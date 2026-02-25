import os
import math
import numpy as np
import argparse
import importlib
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

    model = getattr(importlib.import_module(args.network), 'Net')(n_class=args.n_class).cuda()
    model_cam = getattr(importlib.import_module(args.network), 'Net_CAM')(n_class=args.n_class).cuda()
    model_cam.eval()
    


    loss_weights = None
    
    transform_train = transforms.Compose([transforms.ToTensor()])
    transform_eval = transforms.Compose([transforms.ToTensor()])

    train_dataset = Stage1_TrainDataset(data_path=args.trainroot, transform=transform_train, dataset=args.dataset, img_size=args.img_size)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    


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

    for ep in range(args.max_epoches):
        model.train()
        ep_count = ep_EM = ep_acc = 0
        

        
        for iter, (filename, img, label) in enumerate(train_data_loader):
            img, label= img.cuda(), label.cuda()
            
            # 1. FORWARD RESNET
            out_56, out_28_1, out_28_2, out_deep, y_deep, cam_56, cam_28_1, cam_28_2, cam_deep, feat_56 = model(img)            
            

            loss_cls = (0.1 * F.multilabel_soft_margin_loss(out_56, label, weight=loss_weights) + \
                        0.15 * F.multilabel_soft_margin_loss(out_28_1, label, weight=loss_weights) + \
                        0.25 * F.multilabel_soft_margin_loss(out_28_2, label, weight=loss_weights) + \
                        0.5 * F.multilabel_soft_margin_loss(out_deep, label, weight=loss_weights)) 
            
            loss_adapt_val = torch.tensor(0.0).cuda()
            loss = loss_cls
            
            # Metrics
            prob = y_deep.cpu().data.numpy()
            gt = label.cpu().data.numpy()
            for num, one in enumerate(prob):
                ep_count += 1
                pass_cls = np.where(one > 0.2)[0]
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


        
    os.makedirs(args.save_folder, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.save_folder, 'stage1_cais_last.pth'))


def test_phase(args):
    model = getattr(importlib.import_module(args.network), 'Net_CAM')(n_class=args.n_class)
    model = model.cuda()
    args.weights = os.path.join(args.save_folder, 'stage1_cais_last.pth')
    weights_dict = torch.load(args.weights)
    model.load_state_dict(weights_dict, strict=False)
    model.eval()
    score = infer(model, args.testroot, args.n_class, args, thr=None)
    print(score)

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
    parser.add_argument("--trainroot", default='datasets/LUAD/train/', type=str)
    parser.add_argument("--testroot", default='datasets/LUAD/test/', type=str)
    parser.add_argument("--valroot", default='datasets/LUAD/val/', type=str)

    parser.add_argument("--dataset", default='luad', type=str)
    parser.add_argument("--img_size", default=224, type=int)

    parser.add_argument("--save_folder", default='checkpoints', type=str)

    args = parser.parse_args()

    os.makedirs(args.save_folder, exist_ok=True)
    
    start_time = time.time()
    train_phase(args)
    print(f"Total Training Time: {time.time() - start_time - time_test:.2f}s")
    test_phase(args)