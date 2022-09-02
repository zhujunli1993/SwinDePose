#!/usr/bin/env python3
import os
import glob
import cv2
import torch
import os.path
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import sys 
from tqdm import tqdm
sys.path.append('../..')
from config.common import Config
import pickle as pkl
from utils.basic_utils import Basic_Utils
import scipy.io as scio
import scipy.misc
try:
    from neupeak.utils.webcv2 import imshow, waitKey
except:
    from cv2 import imshow, waitKey
import time
from process import render_mulimage_angles,render_mulimage_signed

config = Config(ds_name='ycb')
bs_utils = Basic_Utils(config)
cls_lst = bs_utils.read_lines(config.ycb_cls_lst_p)

xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])



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

root = 'YCB_Video_Dataset/data'
#root_syn = 'YCB_Video_Dataset/data_syn'
scenes_data = os.listdir(root)
#scenes_syn = os.listdir(root_syn)
file_pa = 'no_pseudo_angles.txt'
listpa = np.loadtxt(file_pa,dtype=str)

#for scene in scenes_data: 0~6000, 6000~12000, 12000~18000
for path in listpa[12000:18000]:
    img_id = path.split('/')[1]
    scene = path.split('/')[0]
    
    #for img_id in os.listdir(os.path.join(root, scene)):
        #if img_id.endswith('box.txt'):
            #img_id = img_id.split('.')[0].split('-')[0]
    
    if os.path.isfile(os.path.join(root, scene, img_id+'-pseudo_angles.png')) and os.path.isfile(os.path.join(root, scene, img_id+'-pseudo_signed.png')):
        continue
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