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
# ren_path = '/workspace/DATA/Linemod_preprocessed/fuse_nrm/phone/file_list.txt'
# contents = np.loadtxt(ren_path, dtype='str')
# for file in contents:
    
#     values = np.load(file)
#     angle = values['angles']
    
#     if len(np.unique(angle)) > 1:

#         print(file)

file = '/workspace/DATA/Linemod_preprocessed/fuse_nrm/phone/7342.npz'
values = np.load(file)
for k in values.keys():
    print(k)

angle = values['angles']
rgb = values['rgb']
depth = values['depth']
K = values['K']
cam_scale = 1000.0
dpt_mm = depth * cam_scale
dpt_mm = dpt_mm.copy().astype(np.uint16)

nrm_map = normalSpeed.depth_normal(dpt_mm, K[0][0], K[1][1], 5, 2000, 20, False)
angle_ = pseudo_nrm_angle(nrm_map)
print('angle: ', np.unique(angle))
print('angle_: ', np.unique(angle_))
rgb_angle = scale_pseudo(angle)
rgb_angle_ = scale_pseudo(angle_)

# depth_ = scale_depth(depth)
img_file = os.path.join('/workspace/DATA/Linemod_preprocessed/data','15','vis_fuse_nrm_7342.png')
cv2.imwrite(img_file, rgb_angle)
img_file = os.path.join('/workspace/DATA/Linemod_preprocessed/data','15','vis_fuse_nrm_7342_.png')
cv2.imwrite(img_file, rgb_angle_)