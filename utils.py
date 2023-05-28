import torch
import torch.nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import math
import json
import os

def get_logger(config, mode='train'):
    folder = os.path.join('logs', config['name'], mode)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return logger.Logger(folder)


def load_config(exp_name):
    """ Loads the configuration file

     Args:
         path: A string indicating the path to the configuration file
     Returns:
         config: A Python dictionary of hyperparameter name-value pairs
         learning rate: The learning rate of the optimzer
         batch_size: Batch size used during training
         num_epochs: Number of epochs to train the network for
         target_classes: A list of strings denoting the classes to
                        build the classifer for
     """
    
    path = os.path.join('./experiments', exp_name, 'config.json')
    with open(path) as file:
        config = json.load(file)
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    max_epochs = config["max_epochs"]

    return config, learning_rate, batch_size, max_epochs

def get_model_name(config, epoch=None):
    """ Generate a name for the model consisting of all the hyperparameter values

    Args:
        name: Name of ckpt
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """

    name = config['name']
    if epoch is None:
        epoch = config['resume_from']

    folder = os.path.join("./experiments", name)         
    if not os.path.exists(folder):
        os.makedirs(folder)

    path = os.path.join(folder, str(epoch)+"epoch")
    return path

def writefile(config, filename, value):
    path = os.path.join('experiments', config['name'], filename)
    with open(path, 'a') as f:
        f.write(value)

def maskFOV_on_BEV(shape, fov=88.0):

    height = shape[0]
    width = shape[1]


    fov = fov / 2

    x = np.arange(width)
    y = np.arange(-height//2, height//2)

    xx, yy = np.meshgrid(x, y)
    angle = np.arctan2(yy, xx) * 180 / np.pi

    in_fov = np.abs(angle) < fov
    in_fov = torch.from_numpy(in_fov.astype(np.float32))

    return in_fov

def get_points_in_a_rotated_box(corners, label_shape=[200, 175]):
    def minY(x0, y0, x1, y1, x):
        if x0 == x1:
            # vertical line, y0 is lowest
            return int(math.floor(y0))

        m = (y1 - y0) / (x1 - x0)

        if m >= 0.0:
            # lowest point is at left edge of pixel column
            return int(math.floor(y0 + m * (x - x0)))
        else:
            # lowest point is at right edge of pixel column
            return int(math.floor(y0 + m * ((x + 1.0) - x0)))


    def maxY(x0, y0, x1, y1, x):
        if x0 == x1:
            # vertical line, y1 is highest
            return int(math.ceil(y1))

        m = (y1 - y0) / (x1 - x0)

        if m >= 0.0:
            # highest point is at right edge of pixel column
            return int(math.ceil(y0 + m * ((x + 1.0) - x0)))
        else:
            # highest point is at left edge of pixel column
            return int(math.ceil(y0 + m * (x - x0)))


    # view_bl, view_tl, view_tr, view_br are the corners of the rectangle
    view = [(corners[i, 0], corners[i, 1]) for i in range(4)]

    pixels = []

    # find l,r,t,b,m1,m2
    l, m1, m2, r = sorted(view, key=lambda p: (p[0], p[1]))
    b, t = sorted([m1, m2], key=lambda p: (p[1], p[0]))

    lx, ly = l
    rx, ry = r
    bx, by = b
    tx, ty = t
    m1x, m1y = m1
    m2x, m2y = m2

    xmin = 0
    ymin = 0
    xmax = label_shape[1]
    ymax = label_shape[0]

    # inward-rounded integer bounds
    # note that we're clamping the area of interest to (xmin,ymin)-(xmax,ymax)
    lxi = max(int(math.ceil(lx)), xmin)
    rxi = min(int(math.floor(rx)), xmax)
    byi = max(int(math.ceil(by)), ymin)
    tyi = min(int(math.floor(ty)), ymax)

    x1 = lxi
    x2 = rxi

    for x in range(x1, x2):
        xf = float(x)

        if xf < m1x:
            # Phase I: left to top and bottom
            y1 = minY(lx, ly, bx, by, xf)
            y2 = maxY(lx, ly, tx, ty, xf)

        elif xf < m2x:
            if m1y < m2y:
                # Phase IIa: left/bottom --> top/right
                y1 = minY(bx, by, rx, ry, xf)
                y2 = maxY(lx, ly, tx, ty, xf)

            else:
                # Phase IIb: left/top --> bottom/right
                y1 = minY(lx, ly, bx, by, xf)
                y2 = maxY(tx, ty, rx, ry, xf)

        else:
            # Phase III: bottom/top --> right
            y1 = minY(bx, by, rx, ry, xf)
            y2 = maxY(tx, ty, rx, ry, xf)

        y1 = max(y1, byi)
        y2 = min(y2, tyi)

        for y in range(y1, y2):
            pixels.append((x, y))

    return pixels

def trasform_label2metric(label, ratio=4, grid_size=0.1, base_height=100):
    '''
    :param label: numpy array of shape [..., 2] of coordinates in label map space
    :return: numpy array of shape [..., 2] of the same coordinates in metric space
    '''

    metric = np.copy(label)
    metric[..., 1] -= base_height
    metric = metric * grid_size * ratio

    return metric