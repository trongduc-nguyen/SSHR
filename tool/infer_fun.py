
import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
from tool import iouutils
from PIL import Image
import torch.nn.functional as F
import os.path
from tool import infer_utils
from tool.GenDataset import Stage1_InferDataset
def _get_amp_dtype(args):
    amp_dtype = getattr(args, "amp_dtype", "none")
    if amp_dtype == "bf16":
        return torch.bfloat16
    if amp_dtype == "fp16":
        return torch.float16
    return None

def _get_class_thresholds(args, thr, n_class):
    if thr is not None:
        return np.full(n_class, float(thr), dtype=np.float32)
    if args.dataset == "bcss":
        if n_class != 4:
            raise ValueError("BCSS inference expects four foreground classes")
        return np.asarray([0.8, 0.9, 0.8, 0.6], dtype=np.float32)
    return np.full(n_class, 0.2 if args.dataset == "luad" else 0.8, dtype=np.float32)

def _tta_transforms():
    return (((), ()), ((3,), (2,)), ((2,), (1,)))

def infer(model, dataroot, n_class, args, thr=None, cam_weights=None):
    model.eval()
    model = model.cuda()
    cam_list = []
    gt_list = []
    infer_dataset = Stage1_InferDataset(
        data_path=os.path.join(dataroot, "img/"),
        img_size=args.img_size,
    )
    infer_data_loader = DataLoader(
        infer_dataset,
        shuffle=False,
        num_workers=getattr(args, "num_workers", 8),
        pin_memory=True,
    )
    if cam_weights is None:
        cam_weights = (0.6, 0.2, 0.2)
    class_thresholds = _get_class_thresholds(args, thr, n_class)
    amp_dtype = _get_amp_dtype(args)

    try:
        with torch.no_grad():
            for _, (img_name_tuple, img_tensor) in enumerate(infer_data_loader):
                img_name = img_name_tuple[0]
                img_path = os.path.join(dataroot, "img/", img_name + ".png")
                orig_img = np.asarray(Image.open(img_path).convert("RGB"))
                orig_img_size = orig_img.shape[:2]
                img_tensor = img_tensor.cuda(non_blocking=True)
                cams_28_1, cams_28_2, cams_deep, probs = [], [], [], []

                for input_flip_dims, cam_flip_dims in _tta_transforms():
                    tta_img = torch.flip(img_tensor, dims=input_flip_dims) if input_flip_dims else img_tensor
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                        _, cam_28_1, cam_28_2, cam_deep, y = model.forward_cam(tta_img)
                        cam_28_1 = F.interpolate(cam_28_1, orig_img_size, mode="bilinear", align_corners=False)[0]
                        cam_28_2 = F.interpolate(cam_28_2, orig_img_size, mode="bilinear", align_corners=False)[0]
                        cam_deep = F.interpolate(cam_deep, orig_img_size, mode="bilinear", align_corners=False)[0]
                    if cam_flip_dims:
                        cam_28_1 = torch.flip(cam_28_1, dims=cam_flip_dims)
                        cam_28_2 = torch.flip(cam_28_2, dims=cam_flip_dims)
                        cam_deep = torch.flip(cam_deep, dims=cam_flip_dims)
                    cams_28_1.append(cam_28_1)
                    cams_28_2.append(cam_28_2)
                    cams_deep.append(cam_deep)
                    probs.append(y)

                c_28_1 = torch.stack(cams_28_1).mean(dim=0)
                c_28_2 = torch.stack(cams_28_2).mean(dim=0)
                c_deep = torch.stack(cams_deep).mean(dim=0)
                prob = torch.stack(probs).mean(dim=0).detach().float().cpu().numpy()[0]
                label = (prob > class_thresholds).astype(np.float32)
                if label.sum() == 0:
                    label[int(np.argmax(prob))] = 1.0

                def norm_np(cam_np):
                    c_min = np.min(cam_np, axis=(1, 2), keepdims=True)
                    c_max = np.max(cam_np, axis=(1, 2), keepdims=True)
                    return (cam_np - c_min) / (c_max - c_min + 1e-8)

                n_28_1 = norm_np(c_28_1.detach().float().cpu().numpy())
                n_28_2 = norm_np(c_28_2.detach().float().cpu().numpy())
                n_deep = norm_np(c_deep.detach().float().cpu().numpy())
                cam = cam_weights[0] * n_28_1 + cam_weights[1] * n_28_2 + cam_weights[2] * n_deep
                cam = cam * label.reshape(n_class, 1, 1)
                cam_dict = infer_utils.cam_npy_to_cam_dict(cam, label)
                cam_score, _ = infer_utils.dict2npy(cam_dict, label, orig_img)
                cam_list.append(infer_utils.cam_npy_to_label_map(cam_score))
                gt_path = os.path.join(dataroot, "mask/", img_name + ".png")
                gt_list.append(np.asarray(Image.open(gt_path)))

        return iouutils.scores(gt_list, cam_list, n_class=n_class)
    except Exception as error:
        print(f"Error: {error}")
        import traceback
        traceback.print_exc()
        return None

