from pyexpat import model

import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
from tool import pyutils, iouutils
from PIL import Image
import torch.nn.functional as F
import os.path
import cv2
from tool import infer_utils
from tool.GenDataset import Stage1_InferDataset
from torchvision import transforms
def infer(model, dataroot, n_class, args, thr=None):
    model.eval()
    model = model.cuda()
    cam_list = []
    gt_list = []
    
    infer_dataset = Stage1_InferDataset(
        data_path=os.path.join(dataroot, 'img/'), 
        img_size=args.img_size    )
    
    infer_data_loader = DataLoader(infer_dataset,
                                   shuffle=False,
                                   num_workers=8,
                                   pin_memory=True)

    if thr is None:
        if args.dataset == 'luad': thr = 0.2
        elif args.dataset == 'bcss': thr = 0.5
            
    try:
        for iter, (img_name_tuple, img_tensor ) in enumerate(infer_data_loader):
            img_name = img_name_tuple[0] 
            
            img_path = os.path.join(dataroot, 'img/', img_name + '.png')
            orig_img = np.asarray(Image.open(img_path))
            orig_img_size = orig_img.shape[:2]

            def _work(i, img, thr=thr):
                with torch.no_grad():
                    img = img.cuda()
                    
                    cam_56, cam_28_1, cam_28_2, cam_deep, y = model.forward_cam(img)
                    
                    y = y.cpu().detach().numpy().tolist()[0]
                    label = torch.tensor([1.0 if j > thr else 0.0 for j in y])
                    
                    c_56 = F.interpolate(cam_56, orig_img_size, mode='bilinear', align_corners=False)[0]
                    c_28_1 = F.interpolate(cam_28_1, orig_img_size, mode='bilinear', align_corners=False)[0]
                    c_28_2 = F.interpolate(cam_28_2, orig_img_size, mode='bilinear', align_corners=False)[0]
                    cam_deep = F.interpolate(cam_deep, orig_img_size, mode='bilinear', align_corners=False)[0]

                    def norm_np(cam_np):
                        c_min = np.min(cam_np, axis=(1, 2), keepdims=True)
                        c_max = np.max(cam_np, axis=(1, 2), keepdims=True)
                        return (cam_np - c_min) / (c_max - c_min + 1e-8)
                    
                    n_56 = norm_np(c_56.cpu().numpy())
                    n_28_1 = norm_np(c_28_1.cpu().numpy())
                    n_28_2 = norm_np(c_28_2.cpu().numpy())
                    cam_deep = norm_np(cam_deep.cpu().numpy())
                    cam =    0.6 * n_28_1 + 0.2 * n_28_2 + 0.2 * cam_deep  
                    cam = cam * label.clone().view(n_class, 1, 1).numpy()
                    return cam, label

            cam_pred_val, label_val = _work(0, img_tensor)
            
            norm_cam = cam_pred_val
            
            cam_dict = infer_utils.cam_npy_to_cam_dict(norm_cam, label_val)
            cam_score, _ = infer_utils.dict2npy(cam_dict, label_val, orig_img)
            seg_map = infer_utils.cam_npy_to_label_map(cam_score)
            
            cam_list.append(seg_map)
            gt_map_path = os.path.join(dataroot, 'mask/', img_name + '.png')
            gt_map = np.array(Image.open(gt_map_path))
            gt_list.append(gt_map)
            
        return iouutils.scores(gt_list, cam_list, n_class=n_class)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()
        return None
def get_mask(model, dataroot, args, save_path):
    if args.dataset == 'luad':
        palette = [0] * 15
        palette[0:3] = [205, 51, 51]
        palette[3:6] = [0, 255, 0]
        palette[6:9] = [65, 105, 225]
        palette[9:12] = [255, 165, 0]
        palette[12:15] = [255, 255, 255]
        thr = 0.31
    elif args.dataset == 'bcss':
        palette = [0] * 15
        palette[0:3] = [255, 0, 0]
        palette[3:6] = [0, 255, 0]
        palette[6:9] = [0, 0, 255]
        palette[9:12] = [153, 0, 255]
        palette[12:15] = [255, 255, 255]
        thr = 0.7
    model.eval()
    transform = transforms.Compose([transforms.ToTensor()])
    infer_dataset = Stage1_InferDataset(data_path=os.path.join(dataroot, 'img/'), transform=transform)
    infer_data_loader = DataLoader(infer_dataset,
                                   shuffle=False,
                                   num_workers=8,
                                   pin_memory=False)
    model = model.cuda()
    for iter, (img_name, img_list) in enumerate(infer_data_loader):
        img_name = img_name[0]
        img_path = os.path.join(dataroot + 'img/' + img_name + '.png')
        orig_img = np.asarray(Image.open(img_path))
        orig_img_size = orig_img.shape[:2]
        def _work(i, img, thr=thr):
            with torch.no_grad():
                img = img.cuda()  
                cam1, cam2, cam3, y = model.forward_cam(img)
                y = y.cpu().detach().numpy().tolist()[0]
                label = torch.tensor([1.0 if j > thr else 0.0 for j in y])
                cam3 = F.interpolate(cam3, orig_img_size, mode='bilinear', align_corners=False)[0]
                cam1 = F.interpolate(cam1, orig_img_size, mode='bilinear', align_corners=False)[0]
                cam2 = F.interpolate(cam2, orig_img_size, mode='bilinear', align_corners=False)[0]
                if args.dataset == 'luad':
                    cam = 0.47 * cam1 + 0.05 * cam2 + 0.47 * cam3
                if args.dataset == 'bcss':
                    cam = 0.11 * cam1 + 0.78 * cam2 + 0.11 * cam3
                cam = cam.cpu().numpy() * label.clone().view(4, 1, 1).numpy()
                return cam, label
        thread_pool = pyutils.BatchThreader(_work, list(enumerate(img_list.unsqueeze(0))),
                                            batch_size=12, prefetch_size=0, processes=8)
        cam_pred = thread_pool.pop_results()
        cams = [pair[0] for pair in cam_pred]
        label = [pair[1] for pair in cam_pred][0]
        sum_cam = np.sum(cams, axis=0)
        norm_cam = (sum_cam - np.min(sum_cam)) / (np.max(sum_cam) - np.min(sum_cam))
        cam_dict = infer_utils.cam_npy_to_cam_dict(norm_cam, label)
        cam_score, bg_score = infer_utils.dict2npy(cam_dict, label, orig_img)
        bgcam_score = np.concatenate((cam_score, bg_score), axis=0)
        seg_map = infer_utils.cam_npy_to_label_map(bgcam_score)
        visualimg = Image.fromarray(seg_map.astype(np.uint8), "P")
        visualimg.putpalette(palette)
        visualimg.save(os.path.join(save_path, img_name + '.png'), format='PNG')
        if iter % 100 == 0:
            print(iter)


