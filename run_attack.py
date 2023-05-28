import os
from tqdm import tqdm
import argparse
import numpy as np
import sys
root_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from utils import get_model_name, load_config
from model import PIXOR
from datagen import get_data_loader
from attack_lib.attack_add_lib import PointAddAttacker
from attack_lib.attack_perturb_lib import PointPerturbAttacker


def build_model(config, device, train=True):
    net = PIXOR(config['geometry'], config['use_bn'])

    if torch.cuda.device_count() <= 1:
        config['mGPUs'] = False
    if config['mGPUs']:
        print("using multi gpu")
        net = nn.DataParallel(net)

    net = net.to(device)
    if not train:
        return net

    return net

def attack(loader, point_idx, save_path):
    print(f"---------begin attack cloud index {point_idx}--------------------")
    raw_lidar_points, label_map = loader.dataset[point_idx]
    raw_lidar_points = torch.from_numpy(raw_lidar_points)
    with torch.no_grad():
        pc = raw_lidar_points.float().cuda(non_blocking=True)
    # attack!
    attacker.attack(pc, save_path)
    return 


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--attack_type', type=str, required=True, help='attack type, perturb or add')
    parser.add_argument('--point_idx', type=int, default=10, help='index of the target point cloud')
    parser.add_argument('--iter_num', type=int, default=2000, help='maximum iteration number')
    parser.add_argument('--attack_lr', type=float, default=0.01, help='attack learning rate')
    parser.add_argument('--save_path', type=str, default="./attack_results/", help='attack results save path')
    args = parser.parse_args()
    
    cudnn.benchmark = True

    exp_name = "default"
    device = torch.device("cuda")
    
    #build models
    config, _, _, _ = load_config(exp_name)
    net = build_model(config, device, train=False)
    net.load_state_dict(torch.load(get_model_name(config), map_location=device))
    net.set_decode(True)
    net.eval()
    train_loader, val_loader = get_data_loader(1, config['use_npy'], geometry=config['geometry'],
                                               frame_range=config['frame_range'])
    
    if args.attack_type == "add":
        attacker = PointAddAttacker(net, config, args.attack_lr, args.iter_num)
    elif args.attack_type == "perturb":
        attacker = PointPerturbAttacker(net, config, args.attack_lr, args.iter_num)
    else:
        print("only support perturb and add")
    
    # run attack
    attack(train_loader, args.point_idx, args.save_path)

