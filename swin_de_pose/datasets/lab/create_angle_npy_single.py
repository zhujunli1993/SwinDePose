import numpy as np
import math
from PIL import Image
import yaml
import os
import cv2
import normalSpeed
import time
from .utils import fill_in_fast, fill_in_multiscale
import torch


def fill_missing(
            dpt, cam_scale, scale_2_80m, fill_type='multiscale',
            extrapolate=False, show_process=False, blur_type='bilateral'
    ):
        dpt = dpt / cam_scale * scale_2_80m
        projected_depth = dpt.copy()
        if fill_type == 'fast':
            final_dpt = fill_in_fast(
                projected_depth, extrapolate=extrapolate, blur_type=blur_type,
                # max_depth=2.0
            )
        elif fill_type == 'multiscale':
            final_dpt, process_dict = fill_in_multiscale(
                projected_depth, extrapolate=extrapolate, blur_type=blur_type,
                show_process=show_process,
                max_depth=3.0
            )
        else:
            raise ValueError('Invalid fill_type {}'.format(fill_type))
        dpt = final_dpt / scale_2_80m * cam_scale
        return dpt
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

def pseudo_nrm_angle(nrm_map, device):
    if device:
        device = torch.device(device)
        nrm_map = torch.from_numpy(nrm_map).to(device)   # 480, 640, 3
    else:
        nrm_map = torch.from_numpy(nrm_map)
    

    # start_time = time.time()
    height=480
    width=640
    
    up_axis = torch.tensor([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0]]).to(device)    # 3, 3

    values = torch.matmul(nrm_map, up_axis)
    
    # Replace (0, 0, 0) in values to (1, 1, 1)  (360, 360, 360)
    values_zeros = torch.zeros(nrm_map.size()[0], nrm_map.size()[1]).to(device)
    values_ones = torch.ones(nrm_map.size()[0], nrm_map.size()[1]).to(device)
    
    new_tensor = torch.where(torch.sum(nrm_map, axis=-1) == 0, values_ones, values_zeros)
    new_tensor = new_tensor.unsqueeze(-1).repeat(1, 1, 3)

    values = values + new_tensor
    
    # Set threshold
    epsilon = torch.tensor(1E-6).to(device)
    values_ones = torch.ones_like(values).to(device)
    values = torch.where(values > (1.0 - epsilon), values_ones, values)
    values = torch.where(values < (epsilon - 1.0), -values_ones, values)
    
    # Compute angles
    angles = torch.arccos(values)*180.0/math.pi
    angles_fill = torch.ones(nrm_map.size()[0], nrm_map.size()[1]).to(device) * 360
    angles_fill = torch.where(torch.sum(nrm_map, axis=-1) == 0, angles_fill, values_zeros)
    angles_fill = angles_fill.unsqueeze(-1).repeat(1, 1, 3)
    angles = angles + angles_fill
    
    angle_x = angles[:, :, 0]
    angle_y = angles[:, :, 1]
    angle_z = angles[:, :, 2]
    
    
                
    angle_x = torch.reshape(angle_x, [height, width])    
    angle_y = torch.reshape(angle_y, [height, width])
    angle_z = torch.reshape(angle_z, [height, width])       
    
    # for visualizztion     
    if False:
        
        angle_x[angle_x==360] = 255
        angle_x = (angle_x-angle_x[angle_x<255].min())*(254/(angle_x[angle_x<255].max()-angle_x[angle_x<255].min()))
        angle_y[angle_y==360] = 255
        angle_y = (angle_y-angle_y[angle_y<255].min())*(254/(angle_y[angle_y<255].max()-angle_y[angle_y<255].min()))
        angle_z[angle_z==360] = 255
        angle_z = (angle_z-angle_z[angle_z<255].min())*(254/(angle_z[angle_z<255].max()-angle_z[angle_z<255].min()))
        # combine three channels and save to a png image
        new_img_angles = torch.dstack((angle_x, angle_y))
        new_img_angles = torch.dstack((new_img_angles, angle_z))
        new_img_angles=new_img_angles.detach().cpu()
        new_img_angles=new_img_angles.numpy()
        new_img_angles = new_img_angles.astype(np.uint8)
        img_file = os.path.join('/workspace/DATA/LabROS/all_nrm.png')
        cv2.imwrite(img_file,new_img_angles)
        # ori = np.load('/workspace/DATA/Linemod_preprocessed/data/01/pseudo_nrm_angles/0000.npz')
        # pseudo = ori['angles']
        # pseudo[:,:,0][pseudo[:,:,0]==360] = 255
        # pseudo[:,:,0][pseudo[:,:,0]<255] = (pseudo[:,:,0][pseudo[:,:,0]<255]-pseudo[:,:,0][pseudo[:,:,0]<255].min())*(254/(pseudo[:,:,0][pseudo[:,:,0]<255].max()-pseudo[:,:,0][pseudo[:,:,0]<255].min()))
        # pseudo[:,:,1][pseudo[:,:,1]==360] = 255
        # pseudo[:,:,1][pseudo[:,:,1]<255] = (pseudo[:,:,1][pseudo[:,:,1]<255]-pseudo[:,:,1][pseudo[:,:,1]<255].min())*(254/(pseudo[:,:,1][pseudo[:,:,1]<255].max()-pseudo[:,:,1][pseudo[:,:,1]<255].min()))
        # pseudo[:,:,2][pseudo[:,:,2]==360] = 255
        # pseudo[:,:,2][pseudo[:,:,2]<255] = (pseudo[:,:,2][pseudo[:,:,2]<255]-pseudo[:,:,2][pseudo[:,:,2]<255].min())*(254/(pseudo[:,:,2][pseudo[:,:,2]<255].max()-pseudo[:,:,2][pseudo[:,:,2]<255].min()))
        # img_file = os.path.join('ori_new.png')
        # cv2.imwrite(img_file,pseudo)
    else:
        new_img_angles = torch.dstack((angle_x, angle_y))
        new_img_angles = torch.dstack((new_img_angles, angle_z))
    # print("running time: ",time.time()-start_time)
    return new_img_angles     


def depth2show(depth):
    show_depth = (depth / depth.max() * 256).astype("uint8")
    return show_depth
       


def single_input(K, cam_scale, depth_path, device):

    K = K
    cam_scale = cam_scale
    dpt_m = np.load(depth_path)
    dpt_m[np.isnan(dpt_m)] = 0.0
    dpt_mm = dpt_m * cam_scale
    dpt_mm = fill_missing(dpt_mm, cam_scale, 1)
    dpt_mm_nrm = dpt_mm.copy().astype(np.uint16)
    nrm_map = normalSpeed.depth_normal(dpt_mm_nrm, K[0][0], K[1][1], 5, 2000, 20, False)
    
    if False:
        show_nrm = depth2show(nrm_map)
        img_file = os.path.join('/workspace/DATA/LabROS/all_normal.png')
        cv2.imwrite(img_file,show_nrm)
        
    nrm_img = pseudo_nrm_angle(nrm_map, device)
    return nrm_img, dpt_m, dpt_mm, nrm_map

def main():
    depth_path = '/workspace/DATA/LabROS/all_depth.npy'
    f_X = 528.1860799146633 # x-focal length
    f_Y = 520.4914794103258 # y-focal length
    c_X = 324.6116067028507 # x coordinate of the optical center
    c_Y = 234.4936855757668 # y coordinate of the optical center
    K = np.array([[f_X, 0, c_X],[0, f_Y, c_Y],[0, 0, 1]])
    cam_scale = 1000.0
    
    nrm_img,_,_,_ = single_input(K, cam_scale, depth_path, device='cuda:0')

if __name__ == "__main__":
    main()