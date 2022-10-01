#!/usr/bin/env python3
from PIL import Image
import numpy as np
import pickle as pkl
import math
import cv2
import yaml

# For pseudo-generate testing.
def dpt_2_pcld(dpt, cam_scale, K):
    xmap = np.array([[j for i in range(640)] for j in range(480)])
    ymap = np.array([[i for i in range(640)] for j in range(480)])
    if len(dpt.shape) > 2:
        dpt = dpt[:, :, 0]
    dpt = dpt.astype(np.float32) / cam_scale
    msk = (dpt > 1e-8).astype(np.float32)
    row = (ymap - K[0][2]) * dpt / K[0][0]
    col = (xmap - K[1][2]) * dpt / K[1][1]
    dpt_3d = np.concatenate(
        (row[..., None], col[..., None], dpt[..., None]), axis=2
    )
    dpt_3d = dpt_3d * msk[:, :, None]
    return dpt_3d

cam_scale = 1000.0
item_name = '/workspace/DATA/Linemod_preprocessed/data/15/depth/0000.png'
if "pkl" in item_name:
    data = pkl.load(open(item_name, "rb"))
    dpt_mm = data['depth'] * 1000.
    K = data['K']
    RT = data['RT']
else:
    with Image.open(item_name) as di:
        dpt_mm = np.array(di)
    meta_file = open('/workspace/DATA/Linemod_preprocessed/data/15/gt.yml', "r")
    meta_lst = yaml.safe_load(meta_file)
    meta = meta_lst[int('0000')]
    meta = meta[0]
    
    R = np.resize(np.array(meta['cam_R_m2c']), (3, 3))
    T = np.array(meta['cam_t_m2c']) / 1000.0
    RT = np.concatenate((R, T[:, None]), axis=1)
    K = np.array([[572.4114, 0.,         325.2611],
                    [0.,        573.57043,  242.04899],
                    [0.,        0.,         1.]])

h, w = dpt_mm.shape 
dpt_m = dpt_mm.astype(np.float32) / cam_scale
dpt_xyz = dpt_2_pcld(dpt_m, 1.0, K)
dpt_xyz[np.isnan(dpt_xyz)] = 0.0
dpt_xyz[np.isinf(dpt_xyz)] = 0.0
dpt_xyz = np.reshape(dpt_xyz,(480, 640, 3))

#partial_path = os.path.join(root, cls_id, str(idx)+'.ptx') 


x_up = np.array([1.0, 0.0, 0.0])
y_up = np.array([0.0, 1.0, 0.0])
z_up = np.array([0.0, 0.0, 1.0])

angle_x = []
signed_x = []
angle_y = []
signed_y = []
angle_z = []
signed_z = []

for i in range(0, h):
    for j in range(1,w):
        p_2 = dpt_xyz[i, j]
        p_1 = dpt_xyz[i, j-1]
        if p_2[0]+p_2[1]+p_2[2]==0.0 or p_1[0]+p_1[1]+p_1[2]==0.0:
            angle_x.append(360.0)
            angle_y.append(360.0)
            angle_z.append(360.0)
            signed_x.append(360.0)
            signed_y.append(360.0)
            signed_z.append(360.0)
            continue
        else:
            difference = p_2 - p_1
            difference_lengh = np.sqrt(math.pow(difference[0],2)+math.pow(difference[1],2)+math.pow(difference[2],2))
            epsilon = 1E-6
            if difference_lengh < epsilon:
                angle_x.append(360.0)
                angle_y.append(360.0)
                angle_z.append(360.0)
                signed_x.append(360.0)
                signed_y.append(360.0)
                signed_z.append(360.0)
                continue
            else:
                value_x = (difference[0]*x_up[0] + difference[1]*x_up[1] + difference[2]*x_up[2]) / difference_lengh
                value_y = (difference[0]*y_up[0] + difference[1]*y_up[1] + difference[2]*y_up[2]) / difference_lengh
                value_z = (difference[0]*z_up[0] + difference[1]*z_up[1] + difference[2]*z_up[2]) / difference_lengh
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
                
                if j == 1:
                    signed_x.append(360.0)
                    signed_y.append(360.0)
                    signed_z.append(360.0)
                    continue
                else:
                    
                    p_0 = dpt_xyz[i, j-2]
                
                    if p_0[0]+p_0[1]+p_0[2]==0.0:
                        signed_x.append(360.0)
                        signed_y.append(360.0)
                        signed_z.append(360.0)
                        continue
                    else:
                        dot_prod = difference[0]*(p_1[0]-p_0[0]) + difference[1]*(p_1[1]-p_0[1]) + difference[2]*(p_1[2]-p_0[2])
                        if dot_prod >= 0.0:
                            signed_x.append(math.acos(value_x)*180.0/math.pi)
                        else:
                            signed_x.append(-1*math.acos(value_x)*180.0/math.pi)
                        if dot_prod >= 0.0:
                            signed_y.append(math.acos(value_y)*180.0/math.pi)
                        else:
                            signed_y.append(-1*math.acos(value_y)*180.0/math.pi)
                        if dot_prod >= 0.0:
                            signed_z.append(math.acos(value_z)*180.0/math.pi)
                        else:
                            signed_z.append(-1*math.acos(value_z)*180.0/math.pi)
    angle_x.append(360.0)
    angle_y.append(360.0)
    angle_z.append(360.0)
    signed_x.append(360.0)
    signed_y.append(360.0)
    signed_z.append(360.0)
angle_x = np.reshape(angle_x, [480, 640])
signed_x = np.reshape(signed_x, [480, 640])
angle_y = np.reshape(angle_y, [480, 640])
signed_y = np.reshape(signed_y, [480, 640])
angle_z = np.reshape(angle_z, [480, 640])
signed_z = np.reshape(signed_z, [480, 640])


angle_x[angle_x==360] = 255
angle_x = (angle_x-angle_x[angle_x<255].min())*(254/(angle_x[angle_x<255].max()-angle_x[angle_x<255].min()))
angle_y[angle_y==360] = 255
angle_y = (angle_y-angle_y[angle_y<255].min())*(254/(angle_y[angle_y<255].max()-angle_y[angle_y<255].min()))
angle_z[angle_z==360] = 255
angle_z = (angle_z-angle_z[angle_z<255].min())*(254/(angle_z[angle_z<255].max()-angle_z[angle_z<255].min()))
    
    
# combine three channels and save to a png image
new_img_angles = np.dstack((angle_x, angle_y))
new_img_angles = np.dstack((new_img_angles, angle_z))

cv2.imwrite('angles.png', new_img_angles)

signed_x[signed_x==360] = 255
signed_x = (signed_x-signed_x[signed_x<255].min())*(254/(signed_x[signed_x<255].max()-signed_x[signed_x<255].min()))
signed_y[signed_y==360] = 255
signed_y = (signed_y-signed_y[signed_y<255].min())*(254/(signed_y[signed_y<255].max()-signed_y[signed_y<255].min()))
signed_z[signed_z==360] = 255
signed_z = (signed_z-signed_z[signed_z<255].min())*(254/(signed_z[signed_z<255].max()-signed_z[signed_z<255].min()))
    
    
# combine three channels and save to a png image
new_img_signed = np.dstack((signed_x, signed_y))
new_img_signed = np.dstack((new_img_signed, signed_z))
cv2.imwrite('angles_signed.png', new_img_signed)
# compare difference
# c_angleX = np.loadtxt('/workspace/REPO/pose_estimation/ffb6d/test1_anglesX.txt')

# print(np.nonzero(c_angleX - angle_x))
# print(len(np.nonzero(c_angleX - angle_x)[0]))
# result = np.where(angle_x != 360.0)
# print(result)
    
