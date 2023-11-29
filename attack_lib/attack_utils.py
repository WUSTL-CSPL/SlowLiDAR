import torch
from shapely.geometry import Polygon
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def max_objects_loss(pred, conf_thres, eta=-1):
    cls_pred = pred[0, ...]
    thres_matrix = torch.full(cls_pred.shape, conf_thres).cuda()
    max_ten = torch.clamp(thres_matrix - cls_pred, min=eta)
    loss = torch.mean(max_ten)
    return loss


def box_overlap_loss(corners, eta=-1):

    if len(corners) == 0 or len(corners) == 1:
        return 0

    overlap_sum = 0
    for i in range(corners.shape[0]):
        x_i1 = corners[i,0]
        x_i2 = corners[i,1]
        x_i3 = corners[i,2]
        x_i4 = corners[i,3]
        x_i_center = (x_i1 + x_i2 + x_i3 + x_i4)/4
        area_i = torch.linalg.norm(x_i1-x_i2, ord=2) * torch.linalg.norm(x_i1-x_i3, ord=2)
        for j in range(i+1, corners.shape[0]):
            x_j1 = corners[j,0]
            x_j2 = corners[j,1]
            x_j3 = corners[j,2]
            x_j4 = corners[j,3]
            x_j_center = (x_j1 + x_j2 + x_j3 + x_j4)/4
            area_j = torch.linalg.norm(x_j1-x_j2, ord=2) * torch.linalg.norm(x_j1-x_j3, ord=2)
            dist_ij = torch.linalg.norm(x_i_center-x_j_center, ord=2)
            max_ten = torch.clamp(torch.sqrt(area_i)*torch.sqrt(area_j)/dist_ij, min=eta)
            overlap_sum += max_ten            

    nums = corners.shape[0] * (corners.shape[0] - 1) * 0.5
    loss = overlap_sum/nums

    return loss


def total_loss(pred, corners, cls_threshold):
    max_obj_loss = max_objects_loss(pred, cls_threshold)
    # Generally use max_object_loss is enough to achieve a decent performance
    # Enabling box_bound_loss and chamfer_dist requires a powerful GPU with large memory
    loss = max_obj_loss
    
    return loss

