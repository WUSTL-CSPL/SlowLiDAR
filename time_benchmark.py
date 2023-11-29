import pdb
import time
import torch
import torch.nn.functional as F
import numpy as np
from shapely.geometry import Polygon
import torch.nn as nn
import sys
sys.path.append("./")

from utils import get_model_name, load_config
from model import PIXOR
from processing.postprocessing import compute_iou
from datagen import get_data_loader
from processing.preprocessing import pixor_preprocess
from processing.postprocessing import filter_pred
from run_attack import build_model
import argparse


def eval_gpu_time(eval_type, point_idx, path):
    
    exp_name = "default"
    device = torch.device("cuda")
    config, _, _, _ = load_config(exp_name)
    net = build_model(config, device, train=False)
    net.load_state_dict(torch.load(get_model_name(config), map_location=device))
    net.set_decode(True)
    net.eval()
    train_loader, val_loader = get_data_loader(1, config['use_npy'], geometry=config['geometry'],
                                               frame_range=config['frame_range'])
    
    if eval_type == "ori":
        #get ori data
        raw_lidar_points, label_map = train_loader.dataset[point_idx]
        raw_lidar_points = torch.from_numpy(raw_lidar_points)
        pc = raw_lidar_points.float().cuda().detach()
    
    elif eval_type == "adv":
        pc = torch.load(path).cuda()

    
    point_data_cube = pixor_preprocess(pc)


    for i in range(20):
        
        t1 = time.time()

        # forward passing
        pred = net(point_data_cube.unsqueeze(0))

        forward_time = time.time() - t1
        
        t2 = time.time()
        # nms
        before_corners, _ = filter_pred(pred, config['cls_threshold'], config['nms_iou_threshold'], config['nms_top'])
        nms_time = time.time() - t2

        total_time = time.time() - t1

        print(f"forward_time: {forward_time}, nms_time: {nms_time}, total_time: {total_time}")
    
    torch.cuda.empty_cache()

    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_type', type=str, required=True, help='evaluation type')
    parser.add_argument('--point_idx', type=int, default=10, help='index of the target point cloud')
    parser.add_argument('--path', type=str, help='path for storing adv examples')
    args = parser.parse_args()

    eval_gpu_time(args.eval_type, args.point_idx, args.path)
