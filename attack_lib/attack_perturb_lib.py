import pdb
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from dist_lib.dist_metric import L2Dist, ChamferDist, HausdorffDist
from attack_lib.attack_utils import total_loss
from processing.preprocessing import pixor_preprocess
from processing.postprocessing import filter_pred


class PointPerturbAttacker:

    def __init__(self, model, config, attack_lr, total_itr):

        self.model = model.cuda()
        self.model.eval()
        self.attack_lr = attack_lr
        self.total_iter = total_itr
        self.chamfer_dist = ChamferDist(method='adv2ori')
        self.cls_threshold = config['cls_threshold']
        self.nms_iou_threshold = config['nms_iou_threshold']
        self.nms_top = config['nms_top']

    def attack(self, raw_lidar_points, save_path):
        
        lidar_points = raw_lidar_points.float().cuda().detach()
        ori_data = lidar_points.clone().detach()
        ori_data.requires_grad = False
        
        adv_data = ori_data.clone().detach() + \
            torch.randn((lidar_points.shape[0], 4)).cuda() * 1e-7
        adv_data.requires_grad_()
        opt = optim.Adam([adv_data], lr=self.attack_lr, weight_decay=0.)
        
        for iteration in range(self.total_iter):

            point_data_cube = pixor_preprocess(adv_data)
            
            t1 = time.time()
            # forward passing
            pred = self.model(point_data_cube.unsqueeze(0))
            forward_time = time.time() - t1
            
            t2 = time.time()
            before_corners, after_corners = filter_pred(pred, self.cls_threshold, self.nms_iou_threshold, self.nms_top)
            nms_time = time.time() - t2

            # compute loss and backward
            loss = total_loss(pred, before_corners, self.cls_threshold)

            opt.zero_grad()
            loss.backward()
            opt.step()
            
            if iteration % 10 == 0:
                print('iteration {}, '
                        'total_loss: {:.4f}, forward_time: {:.4f}, nms_time: {:.4f}'.
                        format(iteration, loss, forward_time, nms_time))
                
            if iteration > 800 and iteration % 10 == 0:
                raw_save_path = save_path + "raw" + "_" + str(iteration) + ".pt"
                torch.save(adv_data, raw_save_path)
                
        torch.cuda.empty_cache()

        return 
