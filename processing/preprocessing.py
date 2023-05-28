import struct
import numpy as np
import torch

####define axis from the specification of KITTI dataset
x_MIN = 0.0
x_MAX = 70.0
y_MIN = -40.0
y_MAX = 40.0
z_MIN = -2.5
z_MAX = 1
x_DIVISION = 0.1
y_DIVISION = 0.1
z_DIVISION = 0.1

X_SIZE = (int)((x_MAX-x_MIN)/x_DIVISION)
Y_SIZE = (int)((y_MAX-y_MIN)/y_DIVISION)
Z_SIZE = (int)((z_MAX-z_MIN)/z_DIVISION)

def getX(x):
    ret = (x-x_MIN)/x_DIVISION
    return int(ret)

def getY(y):
    ret = (y-y_MIN)/y_DIVISION
    return int(ret)

def getZ(z):
    ret = (z-z_MIN)/z_DIVISION
    return int(ret)

def at3(a, b, c):
    return a * (X_SIZE * (Z_SIZE+1)) + b * (Z_SIZE+1) + c

def at2(a, b): 
    return a * X_SIZE+ b


def pixor_preprocess(lidar_raw_data):
    '''
    Change original LiDAR points to BEV maps
    '''
    point_data_cube = torch.zeros((800,700,36), dtype=torch.float32).cuda()
    density_map = torch.zeros(Y_SIZE*X_SIZE, dtype=torch.int)
    
    for i in range(0, lidar_raw_data.shape[0]):
        
        with torch.no_grad():
            
            if lidar_raw_data[i][0] < x_MIN or lidar_raw_data[i][0] > x_MAX or lidar_raw_data[i][1] < y_MIN or lidar_raw_data[i][1] > y_MAX or lidar_raw_data[i][2] < z_MIN or lidar_raw_data[i][2] > z_MAX:
                continue
            
            X = getX(lidar_raw_data[i][0])
            Y = getY(lidar_raw_data[i][1])
            Z = getZ(lidar_raw_data[i][2])
            
            if X >= X_SIZE:
                X = X_SIZE-1
            if Y >= Y_SIZE:
                Y = Y_SIZE-1
            if Z >= Z_SIZE:
                Z = Z_SIZE-1

            density_map[at2(Y, X)] = density_map[at2(Y, X)] + 1

            pos_X = X * x_DIVISION + x_MIN + x_DIVISION/2
            pos_Y = Y * y_DIVISION + y_MIN + y_DIVISION/2
            pos_Z = Z * z_DIVISION + z_MIN + z_DIVISION/2

        dx = 0.5 - 0.5 * torch.tanh(0.1 * (torch.abs(lidar_raw_data[i][0] - pos_X) - 0.05))
        dy = 0.5 - 0.5 * torch.tanh(0.1 * (torch.abs(lidar_raw_data[i][1] - pos_Y) - 0.05))
        dz = 0.5 - 0.5 * torch.tanh(0.1 * (torch.abs(lidar_raw_data[i][2] - pos_Z) - 0.05))

        point_data_cube[Y][X][Z] = dx * dy * dz
        point_data_cube[Y][X][Z_SIZE] += lidar_raw_data[i][3]

    for y in range(Y_SIZE):
        for x in range(X_SIZE):
            if (density_map[at2(y, x)] > 0):
                point_data_cube[y][x][Z_SIZE] /=  float(density_map[at2(y, x)])
    
    point_data_cube =  point_data_cube.permute(2, 0, 1)

    return point_data_cube