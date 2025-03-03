import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import ttach as tta
import prettytable
import time
import glob
import os
import os.path as osp
import multiprocessing.pool as mpp
import multiprocessing as mp

from train import *

import argparse
from utils.config import Config
from tools.mask_convert import mask_save

def get_args():
    parser = argparse.ArgumentParser('description=Semantic segmentation of remote sensing images')
    parser.add_argument("-c", "--config", type=str, default="configs/logcan.py")
    parser.add_argument("--ckpt", type=str, default="work_dirs/LoGCAN_ResNet50_Loveda/epoch=45.ckpt")
    parser.add_argument("--tta", type=str, default="d4")
    parser.add_argument("--masks_output_dir", default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    cfg = Config.fromfile(args.config)

    if args.masks_output_dir is not None:
        masks_output_dir = args.masks_output_dir
    else:
        masks_output_dir = cfg.exp_name + '/figs'

    model = myTrain.load_from_checkpoint(args.ckpt, cfg = cfg)
    model = model.to('cuda')

    model.eval()

    if args.tta == "lr":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip()
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                tta.Rotate90(angles=[90]),
                tta.Scale(scales=[0.5, 0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False)
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)

    metric_cfg1 = cfg.metric_cfg1
    metric_cfg2 = cfg.metric_cfg2

    test_loader = build_dataloader(cfg.dataset_config, mode='val')
    input = next(iter(test_loader))  # Take only the first batch (1 image)
    
    with torch.no_grad():
        raw_predictions, mask, img_id = model(input[0].cuda(), True), input[1].cuda(), input[2]
        pred = raw_predictions.argmax(dim=1)

        mask_pred = pred[0].cpu().numpy()
        mask_name = str(img_id[0])
        results = [(True, mask_pred, cfg.dataset, masks_output_dir, mask_name)]
    
    if not os.path.exists(masks_output_dir):
        os.makedirs(masks_output_dir)
    print("masks_save_dir: ", masks_output_dir)

    mask_save(results[0])  # Save only one mask
