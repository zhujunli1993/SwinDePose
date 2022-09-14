#!/usr/bin/env python3
import os
import glob
import cv2
import torch
import os.path
from PIL import Image
import sys 
import pickle as pkl
import numpy.ma as ma
import numpy as np
from tqdm import tqdm
from process import render_mulimage_angles,render_mulimage_signed,render_mulimage_signed_pkl,render_mulimage_angles_pkl

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


K = np.array([[572.4114, 0.,         325.2611],
                [0.,        573.57043,  242.04899],
                [0.,        0.,         1.]])
'''root = 'Linemod_preprocessed/data'
scenes_data = os.listdir(root)

for scene in scenes_data:
    if not os.path.exists(os.path.join(root, scene, 'pseudo_angles')):
        os.mkdir(os.path.join(root, scene, 'pseudo_angles'))
    if not os.path.exists(os.path.join(root, scene, 'pseudo_signed')):
        os.mkdir(os.path.join(root, scene, 'pseudo_signed'))
    cam_scale = 1000.0
    imgs_data_train = np.loadtxt(os.path.join(root, scene, 'train.txt'),dtype=str)
    
    for img_id in imgs_data_train:
        img = img_id + '.png'
        
        if os.path.isfile(os.path.join(root, scene, 'pseudo_angles', img)) and os.path.isfile(os.path.join(root, scene, 'pseudo_signed', img)):
            continue
        with Image.open(os.path.join(root, scene, 'depth', img)) as di:
            dpt_mm = np.array(di)
        h, w = dpt_mm.shape
        dpt_m = dpt_mm.astype(np.float32) / cam_scale
        dpt_xyz = dpt_2_pcld(dpt_m, 1.0, K)
        dpt_xyz[np.isnan(dpt_xyz)] = 0.0
        dpt_xyz[np.isinf(dpt_xyz)] = 0.0

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
        path_root = "../../ClassificationProject-master/Applications/ComputeSignedAnglesFromPtxFile/"
        code = path_root + "compute_signed_angles_from_ptx_file"   
        os.system(code+" "+partial_path)
        if not os.path.isfile(os.path.join(root, scene, 'pseudo_angles', img)) :
            render_mulimage_angles(os.path.join(root, scene, 'pseudo_angles'), img_id)
        if not os.path.isfile(os.path.join(root, scene, 'pseudo_signed', img)):
            render_mulimage_signed(os.path.join(root, scene, 'pseudo_signed'), img_id)
        
        if os.path.isfile(partial_path):
            os.system("rm " + partial_path)
    
print('Done Train Dataset!!')
'''
root = 'Linemod_preprocessed/renders' #0:15000; 15000:-1 not complete
cls_data = os.listdir(root)
for cls_id in cls_data:    
    file_list = np.loadtxt(os.path.join(root, cls_id, 'file_list.txt'),dtype=str)
    cam_scale = 1000.0
    for item in file_list:
        idx = item.split('/')[-1].split('.')[0]
        data = pkl.load(open(item, "rb"))

        dpt_mm = data['depth'] * cam_scale
        K = data['K']
        h, w = dpt_mm.shape 
        dpt_m = dpt_mm.astype(np.float32) / cam_scale
        dpt_xyz = dpt_2_pcld(dpt_m, 1.0, K)
        dpt_xyz[np.isnan(dpt_xyz)] = 0.0
        dpt_xyz[np.isinf(dpt_xyz)] = 0.0

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

        partial_path = os.path.join(root, cls_id, str(idx)+'.ptx') 
        
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
        path_root = "../../ClassificationProject-master/Applications/ComputeSignedAnglesFromPtxFile/"
        code = path_root + "compute_signed_angles_from_ptx_file"   
        os.system(code+" "+partial_path)
        
        angles = render_mulimage_angles_pkl(os.path.join(root, cls_id, 'pseudo_angles'), str(idx))
        signed = render_mulimage_signed_pkl(os.path.join(root, cls_id, 'pseudo_signed'), str(idx))

        if os.path.isfile(partial_path):
            os.system("rm " + partial_path)
        
        rgb = data['rgb']
        mask = data['mask']
        
        idx_bg = (mask==0)
        idx_obj = (mask!=0)

        angles_new = np.zeros([rgb.shape[0], rgb.shape[1], 3])
        signed_new = np.zeros([rgb.shape[0], rgb.shape[1], 3])
        
        for i in range(3):
            angles_new[idx_bg,i] = rgb[idx_bg,i]
            angles_new[idx_obj,i] = angles[idx_obj,i]
            signed_new[idx_bg,i] = rgb[idx_bg,i]
            signed_new[idx_obj,i] = signed[idx_obj,i]
        
        data['angles'] = angles_new 
        data['signed'] = signed_new  
        
        with open(os.path.join(root, cls_id, item),"wb") as f:
            pkl.dump(data, f)
print('Done Test Dataset!!')   
'''item = '/workspace/raster_triangle/Linemod_preprocessed/renders/phone/0.pkl'
data = pkl.load(open(item, "rb"))
cv2.imwrite('/workspace/raster_triangle/Linemod_preprocessed/renders/phone/0_r.png', data['rgb'])
cv2.imwrite('/workspace/raster_triangle/Linemod_preprocessed/renders/phone/0_s.png', data['signed'])
cv2.imwrite('/workspace/raster_triangle/Linemod_preprocessed/renders/phone/0_a.png', data['angles'])'''


'''root = 'Linemod_preprocessed/fuse'
cls_data = os.listdir(root)
for cls_id in cls_data:    
    file_list = np.loadtxt(os.path.join(root, cls_id, 'file_list.txt'),dtype=str)
    for item in file_list: # 0:2500, 2500:5000, 5000:7500, 7500:-1 # 0:10000, 10000:20000, 20000:30000, //30000:40000, 40000:50000, 50000:60000, 60000:-1
        
        idx = item.split('/')[6].split('.')[0]
        
        data = pkl.load(open(item, "rb"))
        angles = data['angles'] 
        signed = data['signed']
        rgb = data['rgb']
        mask = data['mask']
        rgb_new = np.zeros([rgb.shape[0], rgb.shape[1], 3])
        idx_bg = (mask==0)
        idx_obj = (mask!=0)
        
        for i in range(3):
            rgb_new[idx_bg,i] = rgb[idx_bg,i]
            rgb_new[idx_obj,i] = angles[idx_obj,i]
        
        data['rgb'] = rgb_new
        with open(item,"wb") as f:
            pkl.dump(data, f)
        print(data)
        cv2.imwrite('Linemod_preprocessed/fuse/ape/3_r_new.png', data['rgb'])



root = 'Linemod_preprocessed/fuse'
cls_data = os.listdir(root)
for cls_id in cls_data:    
    file_list = np.loadtxt(os.path.join(root, cls_id,'file_list.txt'),dtype=str)
    
    for item_full in tqdm(file_list): # 0:2500, 2500:5000, 5000:7500, 7500:-1 # 0:10000, 10000:20000, 20000:30000, //30000:40000, 40000:50000, 50000:60000, 60000:-1
        #idx = item.split('/')[-1].split('.')[0]
        #idx = int(idx)
        
        item = item_full[-1]
        
        idx = int(item.split('.')[0])
        data = pkl.load(open(os.path.join(root, 'ape', item), "rb"))
        
        if item_full[0][0] == 'i':
            
            rgb = data['rgb']
            mask = data['mask']
            angles_new = np.zeros([rgb.shape[0], rgb.shape[1], 3])
            signed_new = np.zeros([rgb.shape[0], rgb.shape[1], 3])
            idx_bg = (mask==0)
            idx_obj = (mask!=0)
            angles = data['angles'] 
            signed = data['signed']
            for i in range(3):
                angles_new[idx_bg,i] = rgb[idx_bg,i]
                angles_new[idx_obj,i] = angles[idx_obj,i]
                signed_new[idx_bg,i] = rgb[idx_bg,i]
                signed_new[idx_obj,i] = signed[idx_obj,i]
            
            data['angles'] = angles_new 
            data['signed'] = signed_new  
            
            with open(os.path.join(root, cls_id, item),"wb") as f:
                pkl.dump(data, f)

        else:
            dpt_mm = data['depth'] * 1000.
            K = data['K']

            cam_scale = 1000.0
            h, w = dpt_mm.shape 
            dpt_m = dpt_mm.astype(np.float32) / cam_scale
            dpt_xyz = dpt_2_pcld(dpt_m, 1.0, K)
            dpt_xyz[np.isnan(dpt_xyz)] = 0.0
            dpt_xyz[np.isinf(dpt_xyz)] = 0.0

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

            partial_path = os.path.join(root, cls_id, str(idx)+'.ptx') 
            
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
            path_root = "../../ClassificationProject-master/Applications/ComputeSignedAnglesFromPtxFile/"
            code = path_root + "compute_signed_angles_from_ptx_file"   
            os.system(code+" "+partial_path)
            
            angles = render_mulimage_angles_pkl(os.path.join(root, cls_id, 'pseudo_angles'), str(idx))
            signed = render_mulimage_signed_pkl(os.path.join(root, cls_id, 'pseudo_signed'), str(idx))

            if os.path.isfile(partial_path):
                os.system("rm " + partial_path)
            
            rgb = data['rgb']
            mask = data['mask']
            angles_new = np.zeros([rgb.shape[0], rgb.shape[1], 3])
            signed_new = np.zeros([rgb.shape[0], rgb.shape[1], 3])
            idx_bg = (mask==0)
            idx_obj = (mask!=0)
            
            for i in range(3):
                angles_new[idx_bg,i] = rgb[idx_bg,i]
                angles_new[idx_obj,i] = angles[idx_obj,i]
                signed_new[idx_bg,i] = rgb[idx_bg,i]
                signed_new[idx_obj,i] = signed[idx_obj,i]
            
            data['angles'] = angles_new 
            data['signed'] = signed_new  
            
            with open(os.path.join(root, cls_id, item),"wb") as f:
                pkl.dump(data, f)
item = 'Linemod_preprocessed/renders/ape/47760.pkl'
data = pkl.load(open(item, "rb"))
if (data['angles'][:, 0] - data['signed'][:, 0]).any()==0:
    print('false')

cv2.imwrite('Linemod_preprocessed/renders/ape/0_r.png', data['rgb'])
cv2.imwrite('Linemod_preprocessed/renders/ape/0_s.png', data['signed'])
cv2.imwrite('Linemod_preprocessed/renders/ape/0_a.png', data['angles'])'''
            

