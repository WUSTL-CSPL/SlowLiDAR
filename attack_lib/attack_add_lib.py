import pdb
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from dist_lib.dist_metric import L2Dist, ChamferDist, HausdorffDist
from attack_lib.attack_utils import total_loss
from processing.preprocessing import pixor_preprocess, x_MIN, x_MAX, y_MIN, y_MAX, z_MIN, z_MAX 
from processing.postprocessing import filter_pred


class PointAddAttacker:

    def __init__(self, model, config, attack_lr, total_itr, num_add=600):

        self.model = model.cuda()
        self.model.eval()
        self.num_add = num_add
        self.attack_lr = attack_lr
        self.total_iter = total_itr
        self.chamfer_dist = ChamferDist(method='adv2ori')
        self.cls_threshold = config['cls_threshold']
        self.nms_iou_threshold = config['nms_iou_threshold']
        self.nms_top = config['nms_top']

    def get_critical_points(self, model, lidar_points, n_step=30, n_probe=3000):
        # the initialization is important for point addition attacks, 
        # if you don't find a suitable AE, try to increase n_steps and n_probe
        lidar_points = lidar_points.cuda()
        lidar_points.requires_grad = False
        
        best_critical_points = torch.zeros((self.num_add, 4)).cuda()
        best_grad = torch.zeros(self.num_add).cuda()
        
        for i in range(n_step):
            
            #randomly sample center points
            x_center = random.randint(x_MIN+10, x_MAX-10)
            y_center = random.randint(y_MIN+10, y_MAX-10)
            z_center = random.randint(int(z_MIN+1), int(z_MAX-1)) * 0.5 

            lidar_x = torch.FloatTensor(n_probe).normal_(mean=x_center,std=5).clamp_(min=x_MIN, max=x_MAX-1)
            lidar_y = torch.FloatTensor(n_probe).normal_(mean=y_center,std=5).clamp_(min=y_MIN, max=y_MAX-1)
            lidar_z = torch.FloatTensor(n_probe).normal_(mean=z_center,std=0.5).clamp_(min=z_MIN, max=z_MAX)

            lidar_ref = torch.FloatTensor(n_probe).uniform_(0, 1)

            probe_points = torch.stack([lidar_x, lidar_y, lidar_z, lidar_ref], dim=1)

            lidar_probe_points = probe_points.detach().clone()
            lidar_probe_points = lidar_probe_points.cuda()
            lidar_probe_points.requires_grad_()

            all_points = torch.vstack((lidar_points,lidar_probe_points))
        
            point_data_cube = pixor_preprocess(all_points)

            # forward passing
            pred = model(point_data_cube.unsqueeze(0))

            before_corners, after_corners = filter_pred(pred, self.cls_threshold, self.nms_iou_threshold, self.nms_top)

            # compute loss and backward
            loss = total_loss(pred, before_corners, self.cls_threshold)
            loss.backward() 

            with torch.no_grad():
                grad = lidar_probe_points.grad.data  
                grad = torch.sum(grad ** 2, dim=1) 
                new_grad = torch.cat((grad, best_grad), dim=0)
                new_concate_points = torch.cat((lidar_probe_points, best_critical_points), dim=0)
                best_grad, idx = new_grad.topk(k=self.num_add)
                best_critical_points = torch.stack([
                    new_concate_points[idx[i], :] for i in range(self.num_add)
                ], dim=0).clone().detach()  
            
        return best_critical_points
        
    def attack(self, raw_lidar_points, save_path):

        ori_data = raw_lidar_points.clone().detach() 
        ori_data.requires_grad = False

        cri_data = self.get_critical_points(self.model, ori_data)       
        cri_data.requires_grad_()
        opt = optim.Adam([cri_data], lr=self.attack_lr, weight_decay=0.)
       
        adv_data = torch.cat([ori_data, cri_data], dim=0)

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
  
            adv_data = torch.cat([ori_data, cri_data], dim=0)
            
            if iteration % 10 == 0:   
                print('iteration {}, '
                        'total_loss: {:.4f}, forward_time: {:.4f}, nms_time: {:.4f}'.
                        format(iteration, loss, forward_time, nms_time))
            
            
            if iteration >= 800 and iteration % 10 == 0:
                rawadd_save_path = save_path + "add_" + "_" + str(iteration) + ".pt"
                torch.save(cri_data, rawadd_save_path)
                raw_save_path = save_path + "total_" + "_" + str(iteration) + ".pt"
                torch.save(adv_data, raw_save_path)
            
        torch.cuda.empty_cache()

        return
