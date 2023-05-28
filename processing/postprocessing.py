import torch
from shapely.geometry import Polygon
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_iou(box, boxes):
    """Calculates IoU of the given box with the array of the given boxes.
    box: a polygon
    boxes: a vector of polygons
    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    iou = [box.intersection(b).area / box.union(b).area for b in boxes]

    return np.array(iou, dtype=np.float32)

def non_max_suppression(boxes, scores, nms_iou_threshold, nms_top):
    """Performs non-maximum suppression and returns indices of kept boxes.
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.

    return an numpy array of the positions of picks 0.3
    """

    new_boxes = boxes.detach().clone().numpy()

    assert new_boxes.shape[0] > 0
    if new_boxes.dtype.kind != "f":
        new_boxes = new_boxes.astype(np.float32)

    polygons = convert_format(new_boxes)


    ## 2.1 keeps the box with top confidence scores
    top = nms_top

    ## 2.2 sorted the scores
    ixs = scores.argsort()[::-1][:top]

    pick = []

    ## 2.3 pick the top and delete the boxes with high IoU
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(polygons[i], polygons[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        remove_ixs = np.where(iou > nms_iou_threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)

    return np.array(pick, dtype=np.int32)


def filter_pred(pred, cls_threshold, nms_iou_threshold, nms_top):
    if len(pred.size()) == 4:
        if pred.size(0) == 1:
            pred.squeeze_(0)
        else:
            raise ValueError("Tensor dimension is not right")

    cls_pred = pred[0, ...]
    
    # 1. filter the box proposals with low confidence scores
    activation = cls_pred > cls_threshold

    num_boxes = int(activation.sum())

    if num_boxes == 0:
        #print("No bounding box found")
        return [], []

    ## 1.1 get the corresponding coordinates
    corners = torch.zeros((num_boxes, 8))
    for i in range(7, 15):
        corners[:, i - 7] = torch.masked_select(pred[i, ...], activation)
    
    corners = corners.view(-1, 4, 2)
    
    new_corners = corners.detach().clone()
    
    ## 1.2 get the corresponding confidences scores
    scores = torch.masked_select(cls_pred, activation).detach().cpu().numpy()

    print("nums of boxes before nms: ", new_corners.shape[0])

    # NMS

    # 2. perform nms on the filter box proposals
    selected_ids = non_max_suppression(corners, scores, nms_iou_threshold, nms_top)
    after_corners = corners[selected_ids]

    scores = scores[selected_ids]

    return new_corners, after_corners


def convert_format(boxes_array):
    """

    :param array: an array of shape [# bboxs, 4, 2]
    :return: a shapely.geometry.Polygon object
    """

    polygons = [Polygon([(box[i, 0], box[i, 1]) for i in range(4)]) for box in boxes_array]
    return np.array(polygons)
