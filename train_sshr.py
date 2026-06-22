import os
import math
import numpy as np
import argparse
import importlib
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tool import pyutils, torchutils
from tool.GenDataset import Stage1_TrainDataset
from tool.infer_fun import infer, get_mask
import time
import random
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

cudnn.enabled = True
time_test = 0

# =========================================================================
# 0. PALETTE & UTILS
# =========================================================================
LUAD_PALETTE = [
    205, 51, 51,   # Class 0: Tumor
    0, 255, 0,     # Class 1: Stroma
    65, 105, 225,  # Class 2: Normal
    255, 165, 0,   # Class 3: Necrosis
    255, 255, 255  # Class 4: Background
]
LUAD_PALETTE += [0] * (256 * 3 - len(LUAD_PALETTE))

BCSS_PALETTE = [
    255, 0, 0,     # Class 0
    0, 255, 0,     # Class 1
    0, 0, 255,     # Class 2
    153, 0, 255,   # Class 3
    255, 255, 255  # Class 4: Background
]
BCSS_PALETTE += [0] * (256 * 3 - len(BCSS_PALETTE))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_checkpoint_path(args):
    return os.path.join(args.save_folder, args.checkpoint_name)

def get_infer_thr(args):
    return args.infer_thr if args.infer_thr is not None else None

def get_cam_weights(args):
    return (args.cam_w_28_1, args.cam_w_28_2, args.cam_w_deep)

def get_loss_weights(args):
    return (args.loss_w_56, args.loss_w_28_1, args.loss_w_28_2, args.loss_w_deep)

def apply_palette(mask_np, dataset='luad'):
    mask_img = Image.fromarray(mask_np.astype(np.uint8))
    if dataset == 'bcss':
        mask_img.putpalette(BCSS_PALETTE)
    else:
        mask_img.putpalette(LUAD_PALETTE)
    return mask_img.convert('RGB')

def overlay_heatmap(img_np, cam_np):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_np), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)

def norm_np(cam_np):
    c_min = np.min(cam_np, axis=(1, 2), keepdims=True)
    c_max = np.max(cam_np, axis=(1, 2), keepdims=True)
    return (cam_np - c_min) / (c_max - c_min + 1e-8)

def spatial_normalize(cam_tensor):
    B_sz, C_sz, H_sz, W_sz = cam_tensor.shape
    cam_flat = cam_tensor.view(B_sz, C_sz, -1)
    c_min = cam_flat.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
    c_max = cam_flat.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
    denominator = (c_max - c_min).clamp_min(1e-5)
    return (cam_tensor - c_min) / denominator




class ClassificationEvalDataset(Dataset):
    def __init__(self, data_root, transform=None, img_size=224, n_class=4, dino_version="dino_v2"):
        self.img_dir = os.path.join(data_root, 'img')
        self.mask_dir = os.path.join(data_root, 'mask')
        self.bg_mask_dir = os.path.join(os.path.dirname(data_root.rstrip('/')), 'bg_mask_test')
        self.dino_feat_dir = os.path.join(os.path.dirname(data_root.rstrip('/')), f'test_feats_{dino_version}_{img_size}')
        
        self.img_size = img_size
        self.n_class = n_class
        self.transform = transform 
        self.ids = [os.path.splitext(f)[0] for f in os.listdir(self.img_dir) if not f.startswith('.')]

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = Image.open(os.path.join(self.img_dir, img_id + '.png')).convert('RGB')
        
        mask_np = np.array(Image.open(os.path.join(self.mask_dir, img_id + '.png')))
        
        bg_mask = Image.open(os.path.join(self.bg_mask_dir, img_id + '.png')).convert('L') if os.path.exists(os.path.join(self.bg_mask_dir, img_id + '.png')) else Image.new('L', (self.img_size, self.img_size), 0)
        
        feat_path = os.path.join(self.dino_feat_dir, img_id + '.pt')
        dino_feat = torch.load(feat_path) if os.path.exists(feat_path) else torch.zeros((196, 384))
        if len(dino_feat.shape) == 3: dino_feat = dino_feat.view(-1, dino_feat.shape[-1])

        if img.size[0] != self.img_size or img.size[1] != self.img_size:
            img = TF.resize(img, [self.img_size, self.img_size], interpolation=InterpolationMode.BILINEAR)
            bg_mask = TF.resize(bg_mask, [self.img_size, self.img_size], interpolation=InterpolationMode.NEAREST)
            
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        bg_mask_tensor = (torch.from_numpy(np.array(bg_mask)).long() > 128).float()
        
        label_tensor = torch.zeros(self.n_class)
        for c in np.unique(mask_np):
            if c < self.n_class: label_tensor[c] = 1.0

        return img, label_tensor, bg_mask_tensor, dino_feat

    def __len__(self): return len(self.ids)

def compute_acc(pred_labels, gt_labels):
    pred_correct_count = len(set(pred_labels) & set(gt_labels))
    union = len(gt_labels) + len(pred_labels) - pred_correct_count
    return round(pred_correct_count / union, 4) if union > 0 else 1.0


def train_phase(args):
    global time_test

    set_seed(args.seed)
    model = getattr(importlib.import_module(args.network), 'Net')(n_class=args.n_class).cuda()
    model_cam = getattr(importlib.import_module(args.network), 'Net_CAM')(n_class=args.n_class).cuda()
    model_cam.eval()
    


    loss_weights = None
    
    transform_train = transforms.Compose([transforms.ToTensor()])
    transform_eval = transforms.Compose([transforms.ToTensor()])

    train_dataset = Stage1_TrainDataset(data_path=args.trainroot, transform=transform_train, dataset=args.dataset, img_size=args.img_size)
    data_generator = torch.Generator()
    data_generator.manual_seed(args.seed)
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=data_generator,
    )
    


    max_step = (len(train_dataset) // args.batch_size) * args.max_epoches
    
    param_groups = model.get_parameter_groups()
    optim_params = [
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    ]

    optimizer = torchutils.PolyOptimizer(optim_params, lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)
    
    if args.weights[-7:] == '.params':
        weights_dict = importlib.import_module('network.resnet38d').convert_mxnet_to_torch(args.weights)
        model.load_state_dict(weights_dict, strict=False)
    elif args.weights[-4:] == '.pth':
        model.load_state_dict(torch.load(args.weights), strict=False)
        
    avg_meter = pyutils.AverageMeter('loss_cls', 'loss_adapt', 'avg_ep_EM', 'avg_ep_acc')
    timer = pyutils.Timer("Session started: ")
    best_val_miou = None
    eval_history = []
    os.makedirs(args.save_folder, exist_ok=True)

    for ep in range(args.max_epoches):
        model.train()
        ep_count = ep_EM = ep_acc = 0
        

        
        for iter, (filename, img, label) in enumerate(train_data_loader):
            img, label= img.cuda(), label.cuda()
            
            # 1. FORWARD RESNET
            out_56, out_28_1, out_28_2, out_deep, y_deep, cam_56, cam_28_1, cam_28_2, cam_deep, feat_56 = model(img)            
            

            loss_w_56, loss_w_28_1, loss_w_28_2, loss_w_deep = get_loss_weights(args)
            loss_cls = (loss_w_56 * F.multilabel_soft_margin_loss(out_56, label, weight=loss_weights) + \
                        loss_w_28_1 * F.multilabel_soft_margin_loss(out_28_1, label, weight=loss_weights) + \
                        loss_w_28_2 * F.multilabel_soft_margin_loss(out_28_2, label, weight=loss_weights) + \
                        loss_w_deep * F.multilabel_soft_margin_loss(out_deep, label, weight=loss_weights)) 
            
            loss_adapt_val = torch.tensor(0.0).cuda()
            loss = loss_cls
            
            # Metrics
            prob = y_deep.cpu().data.numpy()
            gt = label.cpu().data.numpy()
            for num, one in enumerate(prob):
                ep_count += 1
                pass_cls = np.where(one > args.train_cls_thr)[0]
                true_cls = np.where(gt[num] == 1)[0]
                if np.array_equal(pass_cls, true_cls): ep_EM += 1
                ep_acc += compute_acc(pass_cls, true_cls)
            
            avg_meter.add({'loss_cls': loss_cls.item(), 'loss_adapt': loss_adapt_val.item()})
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (optimizer.global_step) % 100 == 0 and (optimizer.global_step) != 0:
                timer.update_progress(optimizer.global_step / max_step)
                print('Epoch:%2d' % ep,
                      'Iter:%5d/%5d' % (optimizer.global_step, max_step),
                      'L_Cls:%.4f' % avg_meter.get('loss_cls'),
                      'L_Adpt:%.4f' % avg_meter.get('loss_adapt'),
                      'Tr_EM:%.4f' % round(ep_EM/ep_count, 4),
                      'Tr_Acc:%.4f' % round(ep_acc/ep_count, 4),
                      'lr: %.4f' % optimizer.param_groups[0]['lr'],
                      'Fin:%s' % timer.str_est_finish(), flush=True)


        checkpoint_path = get_checkpoint_path(args)
        if args.save_checkpoints:
            torch.save(model.state_dict(), checkpoint_path)
            if args.save_last_k_checkpoints > 0 and (ep + 1) > args.max_epoches - args.save_last_k_checkpoints:
                epoch_checkpoint_path = os.path.join(args.save_folder, f"stage1_epoch_{ep + 1:04d}.pth")
                torch.save(model.state_dict(), epoch_checkpoint_path)
        if args.eval_every > 0 and ((ep + 1) % args.eval_every == 0):
            state_dict = None if args.save_checkpoints else model.state_dict()
            val_score = test_phase(args, dataroot=args.valroot, split_name='val', checkpoint_path=checkpoint_path, state_dict=state_dict)
            test_score = test_phase(args, dataroot=args.testroot, split_name='test', checkpoint_path=checkpoint_path, state_dict=state_dict)
            val_miou = val_score.get('Mean IoU') if val_score is not None else None
            test_miou = test_score.get('Mean IoU') if test_score is not None else None
            eval_record = {
                'epoch': ep + 1,
                'val_mean_iou': val_miou,
                'test_mean_iou': test_miou,
                'val_mean_dice': val_score.get('Mean Dice') if val_score is not None else None,
                'test_mean_dice': test_score.get('Mean Dice') if test_score is not None else None,
            }
            eval_history.append(eval_record)
            print('[Eval]', json.dumps(eval_record, sort_keys=True), flush=True)
            if val_miou is not None and (best_val_miou is None or val_miou > best_val_miou):
                best_val_miou = val_miou

    return {'best_val_miou': best_val_miou, 'eval_history': eval_history}


def test_phase(args, dataroot=None, split_name='test', checkpoint_path=None, state_dict=None):
    model = getattr(importlib.import_module(args.network), 'Net_CAM')(n_class=args.n_class)
    model = model.cuda()
    if dataroot is None:
        dataroot = args.testroot
    if state_dict is None:
        if checkpoint_path is None:
            checkpoint_path = get_checkpoint_path(args)
        weights_dict = torch.load(checkpoint_path)
    else:
        weights_dict = state_dict
    model.load_state_dict(weights_dict, strict=False)
    model.eval()
    score = infer(model, dataroot, args.n_class, args, thr=get_infer_thr(args), cam_weights=get_cam_weights(args))
    print(f'[{split_name}] {score}', flush=True)
    return score

def parse_batch_choices(value):
    if value is None or value == '':
        return None
    return [int(item.strip()) for item in value.split(',') if item.strip()]

def parse_int_choices(value):
    if value is None or value == '':
        return None
    return [int(item.strip()) for item in value.split(',') if item.strip()]

def run_optuna(args):
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Optuna is required for --optuna_trials. Install it with: pip install optuna") from exc

    base_save_folder = args.save_folder
    batch_choices = parse_batch_choices(args.optuna_batch_choices)

    def objective(trial):
        trial_args = argparse.Namespace(**vars(args))
        trial_args.optuna_trials = 0
        trial_args.save_folder = os.path.join(base_save_folder, f'optuna_trial_{trial.number:04d}')
        trial_args.eval_every = max(1, args.eval_every)
        trial_args.save_checkpoints = False

        trial_args.lr = trial.suggest_float('lr', args.optuna_lr_low, args.optuna_lr_high, log=True)
        trial_args.wt_dec = trial.suggest_float('wt_dec', args.optuna_wt_dec_low, args.optuna_wt_dec_high, log=True)
        if batch_choices:
            trial_args.batch_size = trial.suggest_categorical('batch_size', batch_choices)
        img_size_choices = parse_int_choices(args.optuna_img_size_choices)
        if img_size_choices:
            trial_args.img_size = trial.suggest_categorical('img_size', img_size_choices)

        train_cls_thr_high = min(args.optuna_train_cls_thr_high, 0.5) if args.dataset == 'luad' else args.optuna_train_cls_thr_high
        infer_thr_high = min(args.optuna_thr_high, 0.5) if args.dataset == 'luad' else args.optuna_thr_high
        trial_args.train_cls_thr = trial.suggest_float('train_cls_thr', args.optuna_train_cls_thr_low, train_cls_thr_high)
        trial_args.infer_thr = trial.suggest_float('infer_thr', args.optuna_thr_low, infer_thr_high)

        raw_w_28_1 = trial.suggest_float('cam_w_28_1_raw', args.optuna_cam_weight_low, args.optuna_cam_weight_high)
        raw_w_28_2 = trial.suggest_float('cam_w_28_2_raw', args.optuna_cam_weight_low, args.optuna_cam_weight_high)
        raw_w_deep = trial.suggest_float('cam_w_deep_raw', args.optuna_cam_weight_low, args.optuna_cam_weight_high)
        raw_sum = raw_w_28_1 + raw_w_28_2 + raw_w_deep
        trial_args.cam_w_28_1 = raw_w_28_1 / raw_sum
        trial_args.cam_w_28_2 = raw_w_28_2 / raw_sum
        trial_args.cam_w_deep = raw_w_deep / raw_sum

        raw_loss_w_56 = trial.suggest_float('loss_w_56_raw', args.optuna_loss_weight_low, args.optuna_loss_weight_high)
        raw_loss_w_28_1 = trial.suggest_float('loss_w_28_1_raw', args.optuna_loss_weight_low, args.optuna_loss_weight_high)
        raw_loss_w_28_2 = trial.suggest_float('loss_w_28_2_raw', args.optuna_loss_weight_low, args.optuna_loss_weight_high)
        raw_loss_w_deep = trial.suggest_float('loss_w_deep_raw', args.optuna_loss_weight_low, args.optuna_loss_weight_high)
        raw_loss_sum = raw_loss_w_56 + raw_loss_w_28_1 + raw_loss_w_28_2 + raw_loss_w_deep
        trial_args.loss_w_56 = raw_loss_w_56 / raw_loss_sum
        trial_args.loss_w_28_1 = raw_loss_w_28_1 / raw_loss_sum
        trial_args.loss_w_28_2 = raw_loss_w_28_2 / raw_loss_sum
        trial_args.loss_w_deep = raw_loss_w_deep / raw_loss_sum

        result = train_phase(trial_args)
        best_val_miou = result.get('best_val_miou')
        if best_val_miou is None:
            return float('-inf')
        trial.set_user_attr('cam_w_28_1', trial_args.cam_w_28_1)
        trial.set_user_attr('cam_w_28_2', trial_args.cam_w_28_2)
        trial.set_user_attr('cam_w_deep', trial_args.cam_w_deep)
        trial.set_user_attr('loss_w_56', trial_args.loss_w_56)
        trial.set_user_attr('loss_w_28_1', trial_args.loss_w_28_1)
        trial.set_user_attr('loss_w_28_2', trial_args.loss_w_28_2)
        trial.set_user_attr('loss_w_deep', trial_args.loss_w_deep)
        trial.set_user_attr('effective_params', {
            'batch_size': trial_args.batch_size,
            'img_size': trial_args.img_size,
            'lr': trial_args.lr,
            'wt_dec': trial_args.wt_dec,
            'train_cls_thr': trial_args.train_cls_thr,
            'infer_thr': trial_args.infer_thr,
            'cam_w_28_1': trial_args.cam_w_28_1,
            'cam_w_28_2': trial_args.cam_w_28_2,
            'cam_w_deep': trial_args.cam_w_deep,
            'loss_w_56': trial_args.loss_w_56,
            'loss_w_28_1': trial_args.loss_w_28_1,
            'loss_w_28_2': trial_args.loss_w_28_2,
            'loss_w_deep': trial_args.loss_w_deep,
        })
        trial.set_user_attr('eval_history', result.get('eval_history', []))
        trial_log = {
            'trial': trial.number,
            'value': best_val_miou,
            'params': trial.params,
            'user_attrs': trial.user_attrs,
        }
        os.makedirs(base_save_folder, exist_ok=True)
        with open(os.path.join(base_save_folder, 'optuna_trials.jsonl'), 'a') as f:
            f.write(json.dumps(trial_log, sort_keys=True) + '\n')
        print('[Optuna Trial]', json.dumps(trial_log, sort_keys=True), flush=True)
        return best_val_miou

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.optuna_study_name,
        storage=args.optuna_storage,
        direction=args.optuna_direction,
        sampler=sampler,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.optuna_trials, timeout=args.optuna_timeout)

    best = {
        'best_trial': study.best_trial.number,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'best_user_attrs': study.best_trial.user_attrs,
    }
    os.makedirs(base_save_folder, exist_ok=True)
    best_path = os.path.join(base_save_folder, 'optuna_best.json')
    with open(best_path, 'w') as f:
        json.dump(best, f, indent=2, sort_keys=True)
    print('[Optuna Best]', json.dumps(best, sort_keys=True), flush=True)
    print(f'[Optuna] Best trial summary saved to {best_path}', flush=True)
    return study

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--max_epoches", default=25, type=int)
    parser.add_argument("--network", default="network.resnet38_cls", type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--n_class", default=4, type=int)
    parser.add_argument("--weights", default='init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params', type=str)
    parser.add_argument("--trainroot", default='datasets/LUAD-HistoSeg/training/', type=str)
    parser.add_argument("--testroot", default='datasets/LUAD-HistoSeg/test/', type=str)
    parser.add_argument("--valroot", default='datasets/LUAD-HistoSeg/val/', type=str)

    parser.add_argument("--dataset", default='luad', type=str)
    parser.add_argument("--img_size", default=224, type=int)

    parser.add_argument("--save_folder", default='checkpoints', type=str)
    parser.add_argument("--checkpoint_name", default='stage1_last.pth', type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--eval_every", default=1, type=int)
    parser.add_argument("--infer_thr", default=None, type=float)
    parser.add_argument("--cam_w_28_1", default=0.6, type=float)
    parser.add_argument("--cam_w_28_2", default=0.2, type=float)
    parser.add_argument("--cam_w_deep", default=0.2, type=float)
    parser.add_argument("--loss_w_56", default=0.1, type=float)
    parser.add_argument("--loss_w_28_1", default=0.15, type=float)
    parser.add_argument("--loss_w_28_2", default=0.25, type=float)
    parser.add_argument("--loss_w_deep", default=0.5, type=float)
    parser.add_argument("--train_cls_thr", default=0.2, type=float)
    parser.add_argument("--save_checkpoints", "--save-checkpoints", dest="save_checkpoints", action="store_true", default=True)
    parser.add_argument("--no-save_checkpoints", "--no-save-checkpoints", dest="save_checkpoints", action="store_false")
    parser.add_argument("--save_last_k_checkpoints", "--save-last-k-checkpoints", default=0, type=int)

    parser.add_argument("--optuna_trials", default=0, type=int)
    parser.add_argument("--optuna_study_name", default='sshr_optuna', type=str)
    parser.add_argument("--optuna_storage", default=None, type=str)
    parser.add_argument("--optuna_direction", default='maximize', choices=['maximize', 'minimize'])
    parser.add_argument("--optuna_timeout", default=None, type=int)
    parser.add_argument("--optuna_batch_choices", default='20,32,64', type=str)
    parser.add_argument("--optuna_img_size_choices", default='224,256,320', type=str)
    parser.add_argument("--optuna_lr_low", default=1e-5, type=float)
    parser.add_argument("--optuna_lr_high", default=5e-2, type=float)
    parser.add_argument("--optuna_wt_dec_low", default=1e-7, type=float)
    parser.add_argument("--optuna_wt_dec_high", default=1e-2, type=float)
    parser.add_argument("--optuna_thr_low", default=0.05, type=float)
    parser.add_argument("--optuna_thr_high", default=0.5, type=float)
    parser.add_argument("--optuna_train_cls_thr_low", default=0.05, type=float)
    parser.add_argument("--optuna_train_cls_thr_high", default=0.5, type=float)
    parser.add_argument("--optuna_cam_weight_low", default=0.01, type=float)
    parser.add_argument("--optuna_cam_weight_high", default=1.0, type=float)
    parser.add_argument("--optuna_loss_weight_low", default=0.01, type=float)
    parser.add_argument("--optuna_loss_weight_high", default=1.0, type=float)

    args = parser.parse_args()

    os.makedirs(args.save_folder, exist_ok=True)
    
    start_time = time.time()
    if args.optuna_trials > 0:
        run_optuna(args)
    else:
        train_phase(args)
    print(f"Total Training Time: {time.time() - start_time - time_test:.2f}s")
