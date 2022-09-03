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
from utils.basic_utils import Basic_Utils
import scipy.io as scio
import time
from process import render_mulimage_angles,render_mulimage_signed



opt = BaseOptions().parse()
config = Config(ds_name=opt.dataset_name)
bs_utils = Basic_Utils(config)
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
            img_id = img.split('.')[0].split('-')[0]
            if os.path.isfile(os.path.join(root_syn, img_id+'-pseudo_angles.png')) and os.path.isfile(os.path.join(root_syn, img_id+'-pseudo_signed.png')):
                continue
            with Image.open(os.path.join(root_syn, img_id+'-depth.png')) as di:
                dpt_um = np.array(di)
            meta = scio.loadmat(os.path.join(root_syn, img_id+'-meta.mat'))
            
            K = config.intrinsic_matrix['ycb_K2']

            h, w = opt.height, opt.width
            cam_scale = meta['factor_depth'].astype(np.float32)[0][0]

            dpt_um = bs_utils.fill_missing(dpt_um, cam_scale, 1) 
            dpt_mm = (dpt_um.copy()/10).astype(np.uint16)
            dpt_m = dpt_um.astype(np.float32) / cam_scale

            dpt_xyz = dpt_2_pcld(dpt_m, 1.0, K)
            dpt_xyz = np.reshape(dpt_xyz,(h*w, 3))

            new_col = 0.5000*np.ones([h*w,1])
            dpt_xyz = np.append(dpt_xyz, new_col, 1)

            dpt_xyz = torch.from_numpy(dpt_xyz)

            #pcd format preparation
            xCenter = (torch.max(dpt_xyz[:,0]) + torch.min(dpt_xyz[:,0])) / 2.0000
            yCenter = (torch.max(dpt_xyz[:,1]) + torch.min(dpt_xyz[:,1])) / 2.0000
            zCenter = (torch.max(dpt_xyz[:,2]) + torch.min(dpt_xyz[:,2])) / 2.0000
            image_center = np.asarray([xCenter, yCenter, zCenter])
            image_center = np.reshape(image_center, (1,3))

            partial_path = os.path.join(root_syn, img_id+'.ptx') 

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
            if not os.path.isfile(os.path.join(root_syn, img_id+'-pseudo_angles.png')) :
                render_mulimage_angles(root_syn, img_id)
            if not os.path.isfile(os.path.join(root_syn, img_id+'-pseudo_signed.png')):
                render_mulimage_signed(root_syn, img_id)
            
            if os.path.isfile(partial_path):
                os.system("rm " + partial_path)
                
    print('Done Synthetic Dataset!!')


if opt.real:
    root_real = os.path.join(opt.data_root, 'data')
    scenes = os.listdir(root_real)
    
    for scene in scenes:
        img_list = os.listdir(os.path.join(root_real, scene))
        for img in img_list:
            if img.endswith(".mat"):
                
                img_id = img.split('.')[0].split('-')[0]
                if os.path.isfile(os.path.join(root_real, scene, img_id+'-pseudo_angles.png')) and os.path.isfile(os.path.join(root_real, scene, img_id+'-pseudo_signed.png')):
                    continue
                    
                with Image.open(os.path.join(root_real, scene, img_id+'-depth.png')) as di:
                    dpt_um = np.array(di)

                meta = scio.loadmat(os.path.join(root_real, scene, img_id+'-meta.mat'))
                K = config.intrinsic_matrix['ycb_K1']
                h, w = opt.height, opt.width
                
                cam_scale = meta['factor_depth'].astype(np.float32)[0][0]
                dpt_um = bs_utils.fill_missing(dpt_um, cam_scale, 1)
                if opt.rm_outline: 
                    msk_dp = dpt_um > 1e-6
                    choose = msk_dp.flatten().nonzero()[0].astype(np.uint32)
                    choose_2 = np.array([i for i in range(len(choose))])
                    choose = np.array(choose)[choose_2]                    
                    dpt_mm = (dpt_um.copy()/10).astype(np.uint16)
                    dpt_m = dpt_um.astype(np.float32) / cam_scale
                    dpt_xyz = dpt_2_pcld(dpt_m, 1.0, K)
                    dpt_xyz = np.reshape(dpt_xyz,(h*w, 3))
                    cld = dpt_xyz.reshape(-1, 3)[choose, :]
                    
                    new_col = 0.5000*np.ones([h*w,1])
                    dpt_xyz = np.append(cld, new_col, 1)
                    dpt_xyz = torch.from_numpy(dpt_xyz)
                    
                    
                else:
                    dpt_mm = (dpt_um.copy()/10).astype(np.uint16)
                    dpt_m = dpt_um.astype(np.float32) / cam_scale
                    dpt_xyz = dpt_2_pcld(dpt_m, 1.0, K)
                    dpt_xyz = np.reshape(dpt_xyz,(h*w, 3))

                    new_col = 0.5000*np.ones([h*w,1])
                    dpt_xyz = np.append(dpt_xyz, new_col, 1)
                    dpt_xyz = torch.from_numpy(dpt_xyz)
                
                #pcd format preparation
                xCenter = (torch.max(dpt_xyz[:,0]) + torch.min(dpt_xyz[:,0])) / 2.0000
                yCenter = (torch.max(dpt_xyz[:,1]) + torch.min(dpt_xyz[:,1])) / 2.0000
                zCenter = (torch.max(dpt_xyz[:,2]) + torch.min(dpt_xyz[:,2])) / 2.0000
                image_center = np.asarray([xCenter, yCenter, zCenter])
                image_center = np.reshape(image_center, (1,3))
                
                partial_path = os.path.join(root_real, scene, img_id+'.ptx') 

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

                if not os.path.isfile(os.path.join(root_real, scene, img_id+'-pseudo_angles.png')) :
                    render_mulimage_angles(os.path.join(root_real, scene), img_id)
                if not os.path.isfile(os.path.join(root_real, scene, img_id+'-pseudo_signed.png')):
                    render_mulimage_signed(os.path.join(root_real, scene), img_id)

                os.system("rm " + partial_path)
    print('Done Real Dataset!!')
