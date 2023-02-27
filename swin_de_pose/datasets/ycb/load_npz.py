import numpy as np
import cv2 
import matplotlib.pyplot as plt
import os
import normalSpeed
import math 
def scale_pseudo(pseudo): 
    # Scale the pseudo angles and signed angles to image range (0 ~ 255) 
    pseudo[:,:,0][pseudo[:,:,0]==360] = 255
    pseudo[:,:,0][pseudo[:,:,0]<255] = (pseudo[:,:,0][pseudo[:,:,0]<255]-pseudo[:,:,0][pseudo[:,:,0]<255].min())*(254/(pseudo[:,:,0][pseudo[:,:,0]<255].max()-pseudo[:,:,0][pseudo[:,:,0]<255].min()))
    pseudo[:,:,1][pseudo[:,:,1]==360] = 255
    pseudo[:,:,1][pseudo[:,:,1]<255] = (pseudo[:,:,1][pseudo[:,:,1]<255]-pseudo[:,:,1][pseudo[:,:,1]<255].min())*(254/(pseudo[:,:,1][pseudo[:,:,1]<255].max()-pseudo[:,:,1][pseudo[:,:,1]<255].min()))
    pseudo[:,:,2][pseudo[:,:,2]==360] = 255
    pseudo[:,:,2][pseudo[:,:,2]<255] = (pseudo[:,:,2][pseudo[:,:,2]<255]-pseudo[:,:,2][pseudo[:,:,2]<255].min())*(254/(pseudo[:,:,2][pseudo[:,:,2]<255].max()-pseudo[:,:,2][pseudo[:,:,2]<255].min()))
    
    # pseudo[:,:,0][pseudo[:,:,0]==360] = 255
    # pseudo[:,:,0][pseudo[:,:,0]<255] = pseudo[:,:,0][pseudo[:,:,0]<255]*254.0/180.0
    # pseudo[:,:,1][pseudo[:,:,1]==360] = 255
    # pseudo[:,:,1][pseudo[:,:,1]<255] = pseudo[:,:,1][pseudo[:,:,1]<255]*254.0/180.0
    # pseudo[:,:,2][pseudo[:,:,2]==360] = 255
    # pseudo[:,:,2][pseudo[:,:,2]<255] = pseudo[:,:,2][pseudo[:,:,2]<255]*254.0/180.0
    return pseudo
def scale_depth(pseudo): 
    # Scale the pseudo angles and signed angles to image range (0 ~ 255) 
    pseudo = pseudo * (255/(pseudo.max()-pseudo.min()))

    return pseudo
def pseudo_nrm_angle(nrm_map):
    height=480
    width=640
    # Set up-axis 
    x_up = np.array([1.0, 0.0, 0.0])
    y_up = np.array([0.0, 1.0, 0.0])
    z_up = np.array([0.0, 0.0, 1.0])

    angle_x = []
    # signed_x = []
    angle_y = []
    # signed_y = []
    angle_z = []
    # signed_z = []
    
    for i in range(0, height):
        for j in range(0, width):
            if sum(nrm_map[i, j])==0.0:
                angle_x.append(360.0)
                angle_y.append(360.0)
                angle_z.append(360.0)
                continue
            else:
                nrm_c = nrm_map[i, j]
                epsilon = 1E-6
                value_x = nrm_c[0]*x_up[0] + nrm_c[1]*x_up[1] + nrm_c[2]*x_up[2]
                value_y = nrm_c[0]*y_up[0] + nrm_c[1]*y_up[1] + nrm_c[2]*y_up[2]
                value_z = nrm_c[0]*z_up[0] + nrm_c[1]*z_up[1] + nrm_c[2]*z_up[2]
                if value_x > (1.0 - epsilon):
                    value_x = 1.0
                elif value_x < (epsilon - 1.0):
                    value_x = -1.0 
                angle_x.append(np.arccos(value_x)*180.0/math.pi)
                
                if value_y > (1.0 - epsilon):
                    value_y = 1.0
                elif value_y < (epsilon - 1.0):
                    value_y = -1.0 
                angle_y.append(np.arccos(value_y)*180.0/math.pi)
                
                if value_z > (1.0 - epsilon):
                    value_z = 1.0
                elif value_z < (epsilon - 1.0):
                    value_z = -1.0 
                angle_z.append(np.arccos(value_z)*180.0/math.pi)
    angle_x = np.reshape(angle_x, [height, width])
    # signed_x = np.reshape(signed_x, [height, width])
    angle_y = np.reshape(angle_y, [height, width])
    # signed_y = np.reshape(signed_y, [height, width])
    angle_z = np.reshape(angle_z, [height, width])            
    
    new_img_angles = np.dstack((angle_x, angle_y))
    new_img_angles = np.dstack((new_img_angles, angle_z))
    return new_img_angles            
# Looping to check the wrong case
import pdb;pdb.set_trace()
pth = '/workspace/DATA/YCBV/data/0048/000001-pseudo_nrm_angles.npz'
with np.load(pth) as data:
    angles = data['angles']
    # convert angles and signed angles to image range (0~255)
    sed_angles = scale_pseudo(angles)
    img_file ='/workspace/DATA/YCBV/data/0048/000001-pseudo_nrm_angles.png'
    cv2.imwrite(img_file, sed_angles)