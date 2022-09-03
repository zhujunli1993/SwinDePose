#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import torch
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from config.common import Config
from config.options import BaseOptions
import scipy.io as scio
import time
from process import render_mulimage_angles,render_mulimage_signed



opt = BaseOptions().parse()
config = Config(ds_name=opt.dataset_name)
xmap = np.array([[j for i in range(opt.width)] for j in range(opt.height)])
ymap = np.array([[i for i in range(opt.width)] for j in range(opt.height)])



def dpt_2_pcld(dpt, cam_scale, K):
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


if opt.syn:
    root_syn = os.path.join(opt.data_root, 'data_syn')
    img_data = os.listdir(root_syn)
    for img in img_data:
        if img.endswith(".mat"):
            import pdb; pdb.set_trace()
            img_id = img.split('.')[0].split('-')[0]
            # if os.path.isfile(os.path.join(root, scene, img_id+'-pseudo_angles.png')) and os.path.isfile(os.path.join(root, scene, img_id+'-pseudo_signed.png')):
            #     continue
            with Image.open(os.path.join(root, scene, img_id+'-depth.png')) as di:
                dpt_um = np.array(di)
            meta = scio.loadmat(os.path.join(root, scene, img_id+'-meta.mat'))
            with Image.open(os.path.join(root, scene, img_id+'-label.png')) as li:
                labels = np.array(li)
            
            K = config.intrinsic_matrix['ycb_K2']

            h, w = labels.shape
            cam_scale = meta['factor_depth'].astype(np.float32)[0][0]
            msk_dp = dpt_um > 1e-6

            dpt_um = bs_utils.fill_missing(dpt_um, cam_scale, 1)
            msk_dp = dpt_um > 1e-6

            dpt_mm = (dpt_um.copy()/10).astype(np.uint16)

            dpt_m = dpt_um.astype(np.float32) / cam_scale

            dpt_xyz = dpt_2_pcld(dpt_m, 1.0, K)

            xyz_lst = [dpt_xyz.transpose(2, 0, 1)] # c, h, w
            msk_lst = [dpt_xyz[2, :, :] > 1e-8]

            dpt_xyz = np.reshape(dpt_xyz,(480*640, 3))

            new_col = 0.5000*np.ones([480*640,1])
            dpt_xyz = np.append(dpt_xyz, new_col, 1)

            dpt_xyz = torch.from_numpy(dpt_xyz)

            #pcd format preparation
            xCenter = (torch.max(dpt_xyz[:,0]) + torch.min(dpt_xyz[:,0])) / 2.0000
            yCenter = (torch.max(dpt_xyz[:,1]) + torch.min(dpt_xyz[:,1])) / 2.0000
            zCenter = (torch.max(dpt_xyz[:,2]) + torch.min(dpt_xyz[:,2])) / 2.0000
            image_center = np.asarray([xCenter, yCenter, zCenter])
            image_center = np.reshape(image_center, (1,3))

            partial_path = os.path.join(root, scene, img_id+'.ptx') 

            row_col = np.asarray([h,w])
            row_col = np.reshape(row_col, (2,1))
            matrix1 = np.eye(3, dtype=int)
            matrix2 = np.eye(4, dtype=int)
            with open(partial_path, 'wb') as f:
                np.savetxt(f, row_col , fmt='%i', delimiter=' ')
                np.savetxt(f, image_center, fmt='%.4f', delimiter=' ')
                np.savetxt(f, matrix1, fmt='%i', delimiter=' ')
                np.savetxt(f, matrix2, fmt='%i', delimiter=' ')
                np.savetxt(f, dpt_xyz, fmt = '%.4f', delimiter=' ')
            path_root = "/workspace/ClassificationProject-master/Applications/ComputeSignedAnglesFromPtxFile/"
            code = path_root + "compute_signed_angles_from_ptx_file"   
            os.system(code+" "+partial_path)
            if not os.path.isfile(os.path.join(root, scene, img_id+'-pseudo_angles.png')) :
                render_mulimage_angles(os.path.join(root, scene), img_id)
            if not os.path.isfile(os.path.join(root, scene, img_id+'-pseudo_signed.png')):
                render_mulimage_signed(os.path.join(root, scene), img_id)
            
            if os.path.isfile(partial_path):
                os.system("rm " + partial_path)
                
    print('Done Synthetic Dataset!!')
if opt.real:
    root_real = os.path.join(opt.root, 'data')
    scenes_data = os.listdir(root_real)

'''wordList = []
i=0 
while i<len(scenes_syn):
    wordList.append(scenes_syn[i][0:6])
    i+=1
scene = np.unique(wordList)

for i in range(len(scene)):
    img_id = scene[i]
    
    if os.path.isfile(os.path.join(root_syn, img_id+'-pseudo_angles.png')) and os.path.isfile(os.path.join(root_syn, img_id+'-pseudo_signed.png')):
        print('data/'+img_id)
        continue
        
    with Image.open(os.path.join(root_syn, img_id+'-depth.png')) as di:
        dpt_um = np.array(di)

    meta = scio.loadmat(os.path.join(root_syn, img_id+'-meta.mat'))
    
    
    K = config.intrinsic_matrix['ycb_K1']

    h, w = dpt_um.shape
    cam_scale = meta['factor_depth'].astype(np.float32)[0][0]
    msk_dp = dpt_um > 1e-6

    dpt_um = bs_utils.fill_missing(dpt_um, cam_scale, 1)
    msk_dp = dpt_um > 1e-6

    dpt_mm = (dpt_um.copy()/10).astype(np.uint16)

    dpt_m = dpt_um.astype(np.float32) / cam_scale

    dpt_xyz = dpt_2_pcld(dpt_m, 1.0, K)

    xyz_lst = [dpt_xyz.transpose(2, 0, 1)] # c, h, w
    msk_lst = [dpt_xyz[2, :, :] > 1e-8]

    dpt_xyz = np.reshape(dpt_xyz,(480*640, 3))

    new_col = 0.5000*np.ones([480*640,1])
    dpt_xyz = np.append(dpt_xyz, new_col, 1)

    dpt_xyz = torch.from_numpy(dpt_xyz)

    #pcd format preparation
    xCenter = (torch.max(dpt_xyz[:,0]) + torch.min(dpt_xyz[:,0])) / 2.0000
    yCenter = (torch.max(dpt_xyz[:,1]) + torch.min(dpt_xyz[:,1])) / 2.0000
    zCenter = (torch.max(dpt_xyz[:,2]) + torch.min(dpt_xyz[:,2])) / 2.0000
    image_center = np.asarray([xCenter, yCenter, zCenter])
    image_center = np.reshape(image_center, (1,3))

    partial_path = os.path.join(root_syn,  img_id+'.ptx') 

    row_col = np.asarray([h,w])
    row_col = np.reshape(row_col, (2,1))
    matrix1 = np.eye(3, dtype=int)
    matrix2 = np.eye(4, dtype=int)
    with open(partial_path, 'wb') as f:
        np.savetxt(f, row_col , fmt='%i', delimiter=' ')
        np.savetxt(f, image_center, fmt='%.4f', delimiter=' ')
        np.savetxt(f, matrix1, fmt='%i', delimiter=' ')
        np.savetxt(f, matrix2, fmt='%i', delimiter=' ')
        np.savetxt(f, dpt_xyz, fmt = '%.4f', delimiter=' ')
    path_root = "/workspace/ClassificationProject-master/Applications/ComputeSignedAnglesFromPtxFile/"
    code = path_root + "compute_signed_angles_from_ptx_file"   
    os.system(code+" "+partial_path)

    if not os.path.isfile(os.path.join(root_syn, img_id+'-pseudo_angles.png'))  :
                render_mulimage_angles(os.path.join(root_syn), img_id)
    if not os.path.isfile(os.path.join(root_syn, img_id+'-pseudo_signed.png')) :
                render_mulimage_signed(os.path.join(root_syn), img_id)

    os.system("rm " + partial_path)
print('Done Real Dataset!!')
'''