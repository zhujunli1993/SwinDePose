#!/usr/bin/env python3
import os
import cv2
import os.path
import numpy as np
from PIL import Image
from config.common import Config
from config.options import BaseOptions

from utils.basic_utils import Basic_Utils
from .create_angle_npy_single import single_input
from models.RandLA.helper_tool import DataProcessing as DP



# for get depth_filling function
config_fill = Config(ds_name='ycb')
bs_utils_fill = Basic_Utils(config_fill)
class Dataset():

    def __init__(self, dataset_name, cls_type="duck", DEBUG=False):
        
        self.DEBUG = DEBUG
        self.opt = BaseOptions().parse()
        self.config = Config(ds_name='lab', cls_type=self.opt.linemod_cls)
        
        self.K = self.config.intrinsic_matrix["lab"]
        self.dataset_name = dataset_name
        self.xmap = np.array([[j for i in range(self.opt.width)] for j in range(self.opt.height)])
        self.ymap = np.array([[i for i in range(self.opt.width)] for j in range(self.opt.height)])
        
        self.cls_type = cls_type
        
        self.cls_id = self.config.cls_id
        print("cls_id in lm_dataset.py", self.cls_id)
        self.root = self.config.lm_root
        
        self.rng = np.random
        
        self.depth_pth =  self.config.depth_input
        
    

    def real_syn_gen(self, real_ratio=0.3):
        if len(self.rnd_lst+self.fuse_lst) == 0:
            real_ratio = 1.0
        if self.rng.rand() < real_ratio:  # real
            n_imgs = len(self.real_lst)
            idx = self.rng.randint(0, n_imgs)
            pth = self.real_lst[idx]
            return pth
        else:
            if len(self.fuse_lst) > 0 and len(self.rnd_lst) > 0:
                fuse_ratio = 0.4
            elif len(self.fuse_lst) == 0:
                fuse_ratio = 0.
            else:
                fuse_ratio = 1.
            if self.rng.rand() < fuse_ratio:
                idx = self.rng.randint(0, len(self.fuse_lst))
                pth = self.fuse_lst[idx]
            else:
                idx = self.rng.randint(0, len(self.rnd_lst))
                pth = self.rnd_lst[idx]
            return pth

    def real_gen(self):
        n = len(self.real_lst)
        idx = self.rng.randint(0, n)
        item = self.real_lst[idx]
        return item

    def rand_range(self, rng, lo, hi):
        return rng.rand()*(hi-lo)+lo

    def gaussian_noise(self, rng, img, sigma):
        """add gaussian noise of given sigma to image"""
        img = img + rng.randn(*img.shape) * sigma
        img = np.clip(img, 0, 255).astype('uint8')
        return img

    def linear_motion_blur(self, img, angle, length):
        """:param angle: in degree"""
        rad = np.deg2rad(angle)
        dx = np.cos(rad)
        dy = np.sin(rad)
        a = int(max(list(map(abs, (dx, dy)))) * length * 2)
        if a <= 0:
            return img
        kern = np.zeros((a, a))
        cx, cy = a // 2, a // 2
        dx, dy = list(map(int, (dx * length + cx, dy * length + cy)))
        cv2.line(kern, (cx, cy), (dx, dy), 1.0)
        s = kern.sum()
        if s == 0:
            kern[cx, cy] = 1.0
        else:
            kern /= s
        return cv2.filter2D(img, -1, kern)

    def rgb_add_noise(self, img):
        rng = self.rng
        # apply HSV augmentor
        if rng.rand() > 0:
            
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.uint16)
            hsv_img[:, :, 1] = hsv_img[:, :, 1] * self.rand_range(rng, 1-0.25, 1+.25)
            hsv_img[:, :, 2] = hsv_img[:, :, 2] * self.rand_range(rng, 1-.15, 1+.15)
            hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1], 0, 255)
            hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2], 0, 255)
            img = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if rng.rand() > 0.8:  # motion blur
            r_angle = int(rng.rand() * 360)
            r_len = int(rng.rand() * 15) + 1
            img = self.linear_motion_blur(img, r_angle, r_len)

        if rng.rand() > 0.8:
            if rng.rand() > 0.2:
                img = cv2.GaussianBlur(img, (3, 3), rng.rand())
            else:
                img = cv2.GaussianBlur(img, (5, 5), rng.rand())

        return np.clip(img, 0, 255).astype(np.uint8)

    def add_real_back(self, rgb, labels, dpt, dpt_msk):
        real_item = self.real_gen()
        with Image.open(os.path.join(self.cls_root, "depth", real_item+'.png')) as di:
            real_dpt = np.array(di)
        with Image.open(os.path.join(self.cls_root, "mask", real_item+'.png')) as li:
            bk_label = np.array(li)
        bk_label = (bk_label < 255).astype(rgb.dtype)
        # bk_label_3c = np.repeat(bk_label[:, :, None], 3, 2)
        if len(bk_label.shape) > 2:
            bk_label = bk_label[:, :, 0]
        # Add pseudo-background
        with np.load(os.path.join(self.cls_root, "pseudo_nrm_angles/{}.npz".format(real_item))) as data:
            angles = data['angles']
            
            # convert angles and signed angles to image range (0~255)
            sed_angles = self.scale_pseudo(angles)
            
            sed_angles = Image.fromarray(np.uint8(sed_angles))
            
            back = np.array(sed_angles)[:, :, :3] * bk_label[:, :, None]
        
        
        dpt_back = real_dpt.astype(np.float32) * bk_label.astype(np.float32)
        
        if self.rng.rand() < 0.6:
            msk_back = (labels <= 0).astype(rgb.dtype)
            msk_back = np.repeat(msk_back[:, :, None], 3, 2)
            rgb = rgb * (msk_back == 0).astype(rgb.dtype) + back * msk_back
        
        
        dpt = dpt * (dpt_msk > 0).astype(dpt.dtype) + \
            dpt_back * (dpt_msk <= 0).astype(dpt.dtype)
        return rgb, dpt

    def dpt_2_pcld(self, dpt, cam_scale, K):
        if len(dpt.shape) > 2:
            dpt = dpt[:, :, 0]
        dpt = dpt.astype(np.float32) / cam_scale
        msk = (dpt > 1e-8).astype(np.float32)
        row = (self.ymap - K[0][2]) * dpt / K[0][0]
        col = (self.xmap - K[1][2]) * dpt / K[1][1]
        dpt_3d = np.concatenate(
            (row[..., None], col[..., None], dpt[..., None]), axis=2
        )
        dpt_3d = dpt_3d * msk[:, :, None]
        return dpt_3d
    
    def depth2show(self, depth):
        show_depth = (depth / depth.max() * 256).astype(np.uint8)
        return show_depth
        
    
    def scale_pseudo(self, pseudo):
        
        # Scale the pseudo angles and signed angles to image range (0 ~ 255) 
        pseudo[:,:,0][pseudo[:,:,0]==360.] = 255.
        pseudo[:,:,0][pseudo[:,:,0]<255.] = (pseudo[:,:,0][pseudo[:,:,0]<255.]-pseudo[:,:,0][pseudo[:,:,0]<255.].min())*(254./(pseudo[:,:,0][pseudo[:,:,0]<255.].max()-pseudo[:,:,0][pseudo[:,:,0]<255.].min()))
        pseudo[:,:,1][pseudo[:,:,1]==360.] = 255.
        pseudo[:,:,1][pseudo[:,:,1]<255.] = (pseudo[:,:,1][pseudo[:,:,1]<255.]-pseudo[:,:,1][pseudo[:,:,1]<255.].min())*(254./(pseudo[:,:,1][pseudo[:,:,1]<255.].max()-pseudo[:,:,1][pseudo[:,:,1]<255.].min()))
        pseudo[:,:,2][pseudo[:,:,2]==360.] = 255.
        pseudo[:,:,2][pseudo[:,:,2]<255.] = (pseudo[:,:,2][pseudo[:,:,2]<255.]-pseudo[:,:,2][pseudo[:,:,2]<255.].min())*(254./(pseudo[:,:,2][pseudo[:,:,2]<255.].max()-pseudo[:,:,2][pseudo[:,:,2]<255.].min()))
        
        return pseudo

    def get_item(self):
        
        
        cam_scale = 1000.0
        nrm_img, dpt_m, dpt_mm, nrm_map = single_input(self.K, cam_scale, self.depth_pth, device=None) 
        
        # convert angles and signed angles to image range (0~255)
        sed_angles = self.scale_pseudo(nrm_img)
        if False:
            import pdb;pdb.set_trace()
            show_nrm = self.depth2show(nrm_map)
            img_file = os.path.join('/workspace/DATA/LabROS/all_normal.png')
            cv2.imwrite(img_file,show_nrm)
            new_img_angles = sed_angles.cpu().detach().numpy().astype(np.uint8)
            img_file = os.path.join('/workspace/DATA/LabROS/all_nrm.png')
            cv2.imwrite(img_file,new_img_angles)
        nrm_angles = np.float32(sed_angles)
        
        dpt_mm = dpt_mm.copy().astype(np.uint16)

        dpt_xyz = self.dpt_2_pcld(dpt_m, 1.0, self.K)
        dpt_xyz[np.isnan(dpt_xyz)] = 0.0
        dpt_xyz[np.isinf(dpt_xyz)] = 0.0


        msk_dp = dpt_mm > 1e-6
        choose = msk_dp.flatten().nonzero()[0].astype(np.uint32)
        if len(choose) < 400:
            return None
        choose_2 = np.array([i for i in range(len(choose))])
        if len(choose_2) < 400:
            return None
        if len(choose_2) > self.opt.n_sample_points:
            c_mask = np.zeros(len(choose_2), dtype=int)
            c_mask[:self.opt.n_sample_points] = 1
            np.random.shuffle(c_mask)
            choose_2 = choose_2[c_mask.nonzero()]
        else:
            choose_2 = np.pad(choose_2, (0, self.opt.n_sample_points-len(choose_2)), 'wrap')
        choose = np.array(choose)[choose_2]

        sf_idx = np.arange(choose.shape[0])
        np.random.shuffle(sf_idx)
        choose = choose[sf_idx]

        cld = dpt_xyz.reshape(-1, 3)[choose, :]
        nrm_angles_pt = nrm_angles.reshape(-1, 3)[choose, :].astype(np.float32)
        nrm_pt = nrm_map[:, :, :3].reshape(-1, 3)[choose, :]
        choose = np.array([choose])
        
        cld_angle_nrm = np.concatenate((cld, nrm_angles_pt, nrm_pt), axis=1).transpose(1, 0)

        h, w = self.opt.height, self.opt.width
        nrm_angles = np.transpose(nrm_angles, (2, 0, 1)) # hwc2chw

        xyz_lst = [dpt_xyz.transpose(2, 0, 1)]  # c, h, w
        msk_lst = [dpt_xyz[2, :, :] > 1e-8]
        
        for i in range(3):
            scale = pow(2, i+1)

            nh, nw = h // pow(2, i+3), w // pow(2, i+3)
            ys, xs = np.mgrid[:nh, :nw]
            xyz_lst.append(xyz_lst[0][:, ys*scale, xs*scale])
            msk_lst.append(xyz_lst[-1][2, :, :] > 1e-8)
        

        n_ds_layers = 4
        pcld_sub_s_r = [4, 4, 4, 4]
        inputs = {}
        # DownSample stage
        for i in range(n_ds_layers):
            
            nei_idx = DP.knn_search(
                cld[None, ...], cld[None, ...], 16
            ).astype(np.int32).squeeze(0)
            sub_pts = cld[:cld.shape[0] // pcld_sub_s_r[i], :]
            pool_i = nei_idx[:cld.shape[0] // pcld_sub_s_r[i], :]
            up_i = DP.knn_search(
                sub_pts[None, ...], cld[None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs['cld_xyz%d' % i] = cld.astype(np.float32).copy()
            inputs['cld_nei_idx%d' % i] = nei_idx.astype(np.int32).copy()
            
            inputs['cld_sub_idx%d' % i] = pool_i.astype(np.int32).copy()
            inputs['cld_interp_idx%d' % i] = up_i.astype(np.int32).copy()
            
            cld = sub_pts
        

        item_dict = dict(
            nrm_angles=nrm_angles.astype(np.uint8),  # [c, h, w]
            cld_angle_nrm=cld_angle_nrm.astype(np.float32),  # [9, npts]
            choose=choose.astype(np.int32),  # [1, npts]
            dpt_map_m=dpt_m.astype(np.float32),  # [h, w]
        )
        item_dict.update(inputs)
        
        return item_dict

    
    def __len__(self):
        return 1

    def __getitem__(self, idx):

        return self.get_item()

