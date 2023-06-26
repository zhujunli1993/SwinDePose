import numpy as np
import cv2 
import matplotlib.pyplot as plt
import os
import normalSpeed
import math 
import yaml
from PIL import Image
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
def depth2show(depth, norm_type='max'):
    show_depth = (depth / depth.max() * 256).astype("uint8")
    return show_depth
def norm2bgr(norm):
    norm = ((norm + 1.0) * 127).astype("uint8")
    return norm
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
data=np.load('/workspace/DATA/Linemod_preprocessed/data/09/pseudo_nrm_angles/0183.npz')

img=scale_pseudo(data['angles'])
cv2.imwrite(os.path.join('/workspace/REPO/pose_estimation/ffb6d/duck_0183.png'), img)   

dep =  cv2.imread('/workspace/DATA/Linemod_preprocessed/data/09/depth/0183.png')
dep = depth2show(dep[:,:,0])
cv2.imwrite(os.path.join('/workspace/REPO/pose_estimation/ffb6d/duck_dep_0183.png'), dep) 

K=np.array([[572.4114, 0.,         325.2611],
            [0.,        573.57043,  242.04899],
            [0.,        0.,         1.]])
meta_file = open(os.path.join('/workspace/DATA/Linemod_preprocessed/data','09', 'gt.yml'), "r")
meta_lst = yaml.safe_load(meta_file)
with Image.open(os.path.join('/workspace/DATA/Linemod_preprocessed/data','09', "depth/0183.png")) as di:
    dpt_mm = np.array(di)

meta = meta_lst[int('0183')]
meta = meta[0]

cam_scale = 1000.0

dpt_mm = dpt_mm.copy().astype(np.uint16)
dpt_m = dpt_mm.astype(np.float32) / cam_scale
dpt_xyz = dpt_2_pcld(dpt_m, 1.0, K)
dpt_xyz[np.isnan(dpt_xyz)] = 0.0
dpt_xyz[np.isinf(dpt_xyz)] = 0.0

# rgb, rgb_s = pseudo_gen(args, dpt_xyz)
# if args.vis_img:
#     img_file = os.path.join('/workspace/DATA/Linemod_preprocessed/data',args.cls_num,'vis_angles_{}.png'.format('0183'))
#     cv2.imwrite(img_file, rgb)
#     img_file = os.path.join('/workspace/DATA/Linemod_preprocessed/data',args.cls_num,'vis_signed_{}.png'.format('0183'))
#     cv2.imwrite(img_file, rgb_s)
#     print('Please look at /workspace/DATA/Linemod_preprocessed/data/your_class/angles_or_signed_itemname.png')
#     exit()
# rgb_file = os.path.join('/workspace/DATA/Linemod_preprocessed/data',args.cls_num,'pseudo_angles_signed','{}'.format('0183') )

