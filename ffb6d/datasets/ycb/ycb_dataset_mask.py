#!/usr/bin/env python3
import os
import cv2
import torch
import os.path
import numpy as np
import numpy.ma as ma
import torchvision.transforms as transforms
from PIL import Image
from config.common import Config
from config.options import BaseOptions
import pickle as pkl
from utils.basic_utils import Basic_Utils
import scipy.io as scio
import scipy.misc
try:
    from neupeak.utils.webcv2 import imshow, waitKey
except:
    from cv2 import imshow, waitKey
import normalSpeed
from models.RandLA.helper_tool import DataProcessing as DP

opt = BaseOptions().parse()
config = Config(ds_name='ycb')
bs_utils = Basic_Utils(config)


class Dataset():

    def __init__(self, dataset_name, DEBUG=False):
        self.dataset_name = dataset_name
        self.img_width = 480
        self.img_length = 640
        self.debug = DEBUG
        self.xmap = np.array([[j for i in range(self.img_length)] for j in range(self.img_width)])
        self.ymap = np.array([[i for i in range(self.img_length)] for j in range(self.img_width)])
        self.diameters = {}
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.224])
        self.cls_lst = bs_utils.read_lines(config.ycb_cls_lst_p)
        self.obj_dict = {}
        for cls_id, cls in enumerate(self.cls_lst, start=1):
            self.obj_dict[cls] = cls_id
        self.rng = np.random
        if dataset_name == 'train':
            self.add_noise = True
            self.path = config.train_path
            self.all_lst = bs_utils.read_lines(self.path)
            self.minibatch_per_epoch = len(self.all_lst) // opt.mini_batch_size
            self.real_lst = []
            self.syn_lst = []
            for item in self.all_lst:
                if item[:5] == 'data/':
                    self.real_lst.append(item)
                else:
                    self.syn_lst.append(item)
        else:
            self.pp_data = None
            self.add_noise = False
            self.path = config.test_path
            self.all_lst = bs_utils.read_lines(self.path)
        print("{}_dataset_size: ".format(dataset_name), len(self.all_lst))
        self.root = config.ycb_root
        self.sym_cls_ids = [13, 16, 19, 20, 21]
        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        self.minimum_num_pt = 30
        self.num_pt = 1000


        self.cam_cx_1 = 312.9869
        self.cam_cy_1 = 241.3109
        self.cam_fx_1 = 1066.778
        self.cam_fy_1 = 1067.487

        self.cam_cx_2 = 323.7872
        self.cam_cy_2 = 279.6921
        self.cam_fx_2 = 1077.836
        self.cam_fy_2 = 1078.189

        self.xmap = np.array([[j for i in range(self.img_length)] for j in range(self.img_width)])
        self.ymap = np.array([[i for i in range(self.img_length)] for j in range(self.img_width)])


    def real_syn_gen(self):
        
        n = len(self.real_lst)
        idx = self.rng.randint(0, n)
        item = self.real_lst[idx]

        return item

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
            hsv_img[:, :, 1] = hsv_img[:, :, 1] * self.rand_range(rng, 1.25, 1.45)
            hsv_img[:, :, 2] = hsv_img[:, :, 2] * self.rand_range(rng, 1.15, 1.35)
            hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1], 0, 255)
            hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2], 0, 255)
            img = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if rng.rand() > .8:  # sharpen
            kernel = -np.ones((3, 3))
            kernel[1, 1] = rng.rand() * 3 + 9
            kernel /= kernel.sum()
            img = cv2.filter2D(img, -1, kernel)

        if rng.rand() > 0.8:  # motion blur
            r_angle = int(rng.rand() * 360)
            r_len = int(rng.rand() * 15) + 1
            img = self.linear_motion_blur(img, r_angle, r_len)

        if rng.rand() > 0.8:
            if rng.rand() > 0.2:
                img = cv2.GaussianBlur(img, (3, 3), rng.rand())
            else:
                img = cv2.GaussianBlur(img, (5, 5), rng.rand())

        if rng.rand() > 0.2:
            img = self.gaussian_noise(rng, img, rng.randint(15))
        else:
            img = self.gaussian_noise(rng, img, rng.randint(25))

        if rng.rand() > 0.8:
            img = img + np.random.normal(loc=0.0, scale=7.0, size=img.shape)

        return np.clip(img, 0, 255).astype(np.uint8)

    def add_real_back(self, rgb, rgb_s, labels, dpt, dpt_msk):
        real_item = self.real_gen()
        with Image.open(os.path.join(self.root, real_item+'-depth.png')) as di:
            real_dpt = np.array(di)
        with Image.open(os.path.join(self.root, real_item+'-label.png')) as li:
            bk_label = np.array(li)
        bk_label = (bk_label <= 0).astype(rgb.dtype)
        bk_label_3c = np.repeat(bk_label[:, :, None], 3, 2)
        with Image.open(os.path.join(self.root, real_item+'-pseudo_angles.png')) as ri:
            back = np.array(ri)[:, :, :3] * bk_label_3c
        with Image.open(os.path.join(self.root, real_item+'-pseudo_signed.png')) as rs:
            back_s = np.array(rs)[:, :, :3] * bk_label_3c
        dpt_back = real_dpt.astype(np.float32) * bk_label.astype(np.float32)

        msk_back = (labels <= 0).astype(rgb.dtype)
        msk_back = np.repeat(msk_back[:, :, None], 3, 2)
        rgb = rgb * (msk_back == 0).astype(rgb.dtype) + back * msk_back
        rgb_s = rgb_s * (msk_back == 0).astype(rgb_s.dtype) + back_s * msk_back
        dpt = dpt * (dpt_msk > 0).astype(dpt.dtype) + \
            dpt_back * (dpt_msk <= 0).astype(dpt.dtype)

        return rgb, rgb_s, dpt

    def dpt_2_pcld(self, dpt,cam_scale, K):
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

    def get_item(self, item_name):
        
        label_idx = item_name.split(',')[1]
        item_name = item_name.split(',')[0]
        
        with Image.open(os.path.join(self.root, item_name+'-depth.png')) as di:
            dpt_um = np.array(di)
        with Image.open(os.path.join(self.root, item_name+'-label.png')) as li:
            labels = np.array(li) # labels.unique=[0,1,5,7,8,13,18,21]
        
        meta = scio.loadmat(os.path.join(self.root, item_name+'-meta.mat'))
        if item_name[:8] != 'data_syn' and int(item_name[5:9]) >= 60:
            K = config.intrinsic_matrix['ycb_K2']
        else:
            K = config.intrinsic_matrix['ycb_K1']

        with Image.open(os.path.join(self.root, item_name+'-pseudo_angles.png')) as ri:
            if self.add_noise:
                ri = self.trancolor(ri)
            rgb = np.array(ri)[:, :, :3]
        
        with Image.open(os.path.join(self.root, item_name+'-pseudo_signed.png')) as rs:
            if self.add_noise:
                rs = self.trancolor(rs)
            rgb_s = np.array(rs)[:, :, :3]
        cls_id_lst = meta['cls_indexes'].flatten().astype(np.uint32) #[ 1,  5,  7,  8, 13, 18, 21]

        
        # Randomly pick objects from the scene 
        cls_id_label = int(label_idx)
        mask_label = ma.getmaskarray(ma.masked_equal(labels, cls_id_label)) # mask_label.shape=[480,640], mask_label.unique=[true,false]
        
        # RGB and RGB_S cropped image.
        img = np.zeros([self.img_width, self.img_length, 3]).astype(np.float32)
        for i in range(3):
            img[:,:,i] = mask_label * rgb[:,:,i]

        
        img_s = np.zeros([self.img_width, self.img_length, 3]).astype(np.float32)
        for i in range(3):
            img_s[:,:,i] = mask_label * rgb_s[:,:,i]
        
        # Masked labels
        rgb_labels = mask_label * labels
        # Masked depth image.
        dpt_masked = np.zeros([self.img_width, self.img_length]).astype(np.float32)
        dpt_um = mask_label * dpt_um
        
        rnd_typ = 'syn' if 'syn' in item_name else 'real'
        cam_scale = meta['factor_depth'].astype(np.float32)[0][0]
        msk_dp = dpt_um > 1e-6

        if self.add_noise and rnd_typ == 'syn':
            rgb = self.rgb_add_noise(rgb)
            rgb_s = self.rgb_add_noise(rgb_s)
            rgb, rgb_s, dpt_um = self.add_real_back(rgb, rgb_s, dpt_um, msk_dp)
            if self.rng.rand() > 0.8:
                rgb = self.rgb_add_noise(rgb)
                rgb_s = self.rgb_add_noise(rgb_s)

        
        rgb_c = np.concatenate((img, img_s), axis=2)  #[h,w,6]
        dpt_um = bs_utils.fill_missing(dpt_um, cam_scale, 1) #??
        msk_dp = dpt_um > 1e-6

        dpt_mm = (dpt_um.copy()/10).astype(np.uint16)
        nrm_map = normalSpeed.depth_normal(
            dpt_mm, K[0][0], K[1][1], 5, 2000, 20, False
        )
        
        dpt_m = dpt_um.astype(np.float32) / cam_scale
        dpt_xyz = self.dpt_2_pcld(dpt_m, 1.0, K)
        
        choose = msk_dp.flatten().nonzero()[0].astype(np.uint32)
        if len(choose) < 400:
            return None
        choose_2 = np.array([i for i in range(len(choose))])
        if len(choose_2) < 400:
            return None
        
        if len(choose_2) > opt.n_sample_points: # Randomly downsampling
            c_mask = np.zeros(len(choose_2), dtype=int)
            c_mask[:opt.n_sample_points] = 1
            np.random.shuffle(c_mask)
            choose_2 = choose_2[c_mask.nonzero()]
        else:
            choose_2 = np.pad(choose_2, (0, opt.n_sample_points-len(choose_2)), 'wrap')
        choose = np.array(choose)[choose_2]
        
        # Randomly picking 3D points
        sf_idx = np.arange(choose.shape[0])
        np.random.shuffle(sf_idx)
        choose = choose[sf_idx]
        cld = dpt_xyz.reshape(-1, 3)[choose, :]
        rgb_c_pt = rgb_c.reshape(-1, 6)[choose, :].astype(np.float32)
        
        nrm_pt = nrm_map[:, :, :3].reshape(-1, 3)[choose, :]
        labels_pt = rgb_labels.flatten()[choose]
        choose = np.array([choose])
        cld_rgb_nrm = np.concatenate((cld, rgb_c_pt, nrm_pt), axis=1).transpose(1, 0)
        cls_id_lst = meta['cls_indexes'].flatten().astype(np.uint32)
        

        RTs, kp3ds, ctr3ds, cls_ids, kp_targ_ofst, ctr_targ_ofst = self.get_pose_gt_info_obj(
            cld, labels_pt, cls_id_lst, cls_id_label, meta
        )
        h, w = rgb_labels.shape[0],rgb_labels.shape[1]

        dpt_6c = np.concatenate((dpt_xyz, nrm_map[:, :, :3]), axis=2).transpose(2, 0, 1)
        rgb_c = np.transpose(rgb_c, (2, 0, 1)) # hwc2chw
        
        xyz_lst = [dpt_xyz.transpose(2, 0, 1)] # c, h, w

        msk_lst = [dpt_xyz[2, :, :] > 1e-8]

        for i in range(3):
            scale = pow(2, i+1)
            nh, nw = h // pow(2, i+1), w // pow(2, i+1)
            ys, xs = np.mgrid[:nh, :nw]
            xyz_lst.append(xyz_lst[0][:, ys*scale, xs*scale])
            msk_lst.append(xyz_lst[-1][2, :, :] > 1e-8)
        sr2dptxyz = {
            pow(2, ii): item.reshape(3, -1).transpose(1, 0) for ii, item in enumerate(xyz_lst)
        }
        sr2msk = {
            pow(2, ii): item.reshape(-1) for ii, item in enumerate(msk_lst)
        }

        rgb_ds_sr = [4, 8, 8, 8]
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
            inputs['cld_xyz%d'%i] = cld.astype(np.float32).copy()
            inputs['cld_nei_idx%d'%i] = nei_idx.astype(np.int32).copy()
            inputs['cld_sub_idx%d'%i] = pool_i.astype(np.int32).copy()
            inputs['cld_interp_idx%d'%i] = up_i.astype(np.int32).copy()
            nei_r2p = DP.knn_search(
                sr2dptxyz[rgb_ds_sr[i]][None, ...], sub_pts[None, ...], 16
            ).astype(np.int32).squeeze(0)
            inputs['r2p_ds_nei_idx%d'%i] = nei_r2p.copy()
            nei_p2r = DP.knn_search(
                sub_pts[None, ...], sr2dptxyz[rgb_ds_sr[i]][None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs['p2r_ds_nei_idx%d'%i] = nei_p2r.copy()
            cld = sub_pts

        n_up_layers = 3
        rgb_up_sr = [4, 2, 2]
        for i in range(n_up_layers):
            r2p_nei = DP.knn_search(
                sr2dptxyz[rgb_up_sr[i]][None, ...],
                inputs['cld_xyz%d'%(n_ds_layers-i-1)][None, ...], 16
            ).astype(np.int32).squeeze(0)
            inputs['r2p_up_nei_idx%d'%i] = r2p_nei.copy()
            p2r_nei = DP.knn_search(
                inputs['cld_xyz%d'%(n_ds_layers-i-1)][None, ...],
                sr2dptxyz[rgb_up_sr[i]][None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs['p2r_up_nei_idx%d'%i] = p2r_nei.copy()

        show_rgb = rgb.transpose(1, 2, 0).copy()[:, :, ::-1]

        item_dict = dict(
            rgb=rgb_c.astype(np.uint8),  # [c, h, w]
            cld_rgb_nrm=cld_rgb_nrm.astype(np.float32),  # [9, npts]
            choose=choose.astype(np.int32),  # [1, npts]
            labels=labels_pt.astype(np.int32),  # [npts]
            rgb_labels=rgb_labels.astype(np.int32),  # [h, w]
            dpt_map_m=dpt_m.astype(np.float32),  # [h, w]
            RTs=RTs.astype(np.float32),
            kp_targ_ofst=kp_targ_ofst.astype(np.float32),
            ctr_targ_ofst=ctr_targ_ofst.astype(np.float32),
            cls_ids=cls_ids.astype(np.int32),
            ctr_3ds=ctr3ds.astype(np.float32),
            kp_3ds=kp3ds.astype(np.float32),
        )
        item_dict.update(inputs)
        
        return item_dict


    def get_bbox(self, label):
        border_list = self.border_list
        img_width = self.img_width
        img_length = self.img_length
        rows = np.any(label, axis=1)
        cols = np.any(label, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        rmax += 1
        cmax += 1
        r_b = rmax - rmin
        for tt in range(len(border_list)):
            if r_b > border_list[tt] and r_b < border_list[tt + 1]:
                r_b = border_list[tt + 1]
                break
        c_b = cmax - cmin
        for tt in range(len(border_list)):
            if c_b > border_list[tt] and c_b < border_list[tt + 1]:
                c_b = border_list[tt + 1]
                break
        center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
        rmin = center[0] - int(r_b / 2)
        rmax = center[0] + int(r_b / 2)
        cmin = center[1] - int(c_b / 2)
        cmax = center[1] + int(c_b / 2)
        if rmin < 0:
            delt = -rmin
            rmin = 0
            rmax += delt
        if cmin < 0:
            delt = -cmin
            cmin = 0
            cmax += delt
        if rmax > img_width:
            delt = rmax - img_width
            rmax = img_width
            rmin -= delt
        if cmax > img_length:
            delt = cmax - img_length
            cmax = img_length
            cmin -= delt
        return rmin, rmax, cmin, cmax

    def get_pose_gt_info(self, cld, labels, cls_id_lst, meta):
        RTs = np.zeros((config.n_objects, 3, 4))
        kp3ds = np.zeros((config.n_objects, opt.n_keypoints, 3))
        ctr3ds = np.zeros((config.n_objects, 3))
        cls_ids = np.zeros((config.n_objects, 1))
        kp_targ_ofst = np.zeros((opt.n_sample_points, opt.n_keypoints, 3))
        ctr_targ_ofst = np.zeros((opt.n_sample_points, 3))
        for i, cls_id in enumerate(cls_id_lst):
            r = meta['poses'][:, :, i][:, 0:3]
            t = np.array(meta['poses'][:, :, i][:, 3:4].flatten()[:, None])
            RT = np.concatenate((r, t), axis=1)
            RTs[i] = RT

            ctr = bs_utils.get_ctr(self.cls_lst[cls_id-1]).copy()[:, None]
            ctr = np.dot(ctr.T, r.T) + t[:, 0]
            ctr3ds[i, :] = ctr[0]
            msk_idx = np.where(labels == cls_id)[0]

            target_offset = np.array(np.add(cld, -1.0*ctr3ds[i, :]))
            ctr_targ_ofst[msk_idx,:] = target_offset[msk_idx, :]
            cls_ids[i, :] = np.array([cls_id])

            key_kpts = ''
            if opt.n_keypoints == 8:
                kp_type = 'farthest'
            else:
                kp_type = 'farthest{}'.format(opt.n_keypoints)
            kps = bs_utils.get_kps(
                self.cls_lst[cls_id-1], kp_type=kp_type, ds_type='ycb'
            ).copy()
            kps = np.dot(kps, r.T) + t[:, 0]
            kp3ds[i] = kps

            target = []
            for kp in kps:
                target.append(np.add(cld, -1.0*kp))
            target_offset = np.array(target).transpose(1, 0, 2)  # [npts, nkps, c]
            kp_targ_ofst[msk_idx, :, :] = target_offset[msk_idx, :, :]
        return RTs, kp3ds, ctr3ds, cls_ids, kp_targ_ofst, ctr_targ_ofst

    def get_pose_gt_info_obj(self, cld, labels, cls_id_lst, pick_id, meta):
        RTs = np.zeros((1, 3, 4))
        kp3ds = np.zeros((1, opt.n_keypoints, 3))
        ctr3ds = np.zeros((1, 3))
        cls_ids = np.zeros((1, 1))
        kp_targ_ofst = np.zeros((opt.n_sample_points, opt.n_keypoints, 3))
        ctr_targ_ofst = np.zeros((opt.n_sample_points, 3))
        for i, cls_id in enumerate(cls_id_lst):
            if cls_id == pick_id:
                r = meta['poses'][:, :, i][:, 0:3]
                t = np.array(meta['poses'][:, :, i][:, 3:4].flatten()[:, None])
                RTs = np.concatenate((r, t), axis=1)
                #RTs[i] = RT

                ctr = bs_utils.get_ctr(self.cls_lst[cls_id-1]).copy()[:, None]
                ctr = np.dot(ctr.T, r.T) + t[:, 0]
                ctr3ds[0, :] = ctr[0]
                msk_idx = np.where(labels == cls_id)[0]

                target_offset = np.array(np.add(cld, -1.0*ctr3ds[0, :]))
                ctr_targ_ofst[msk_idx,:] = target_offset[msk_idx, :]
                cls_ids[0, :] = np.array([cls_id])

                key_kpts = ''
                if config.n_keypoints == 8:
                    kp_type = 'farthest'
                else:
                    kp_type = 'farthest{}'.format(config.n_keypoints)
                kps = bs_utils.get_kps(
                    self.cls_lst[cls_id-1], kp_type=kp_type, ds_type='ycb'
                ).copy()
                kps = np.dot(kps, r.T) + t[:, 0]
                kp3ds = kps

                target = []
                for kp in kps:
                    target.append(np.add(cld, -1.0*kp))
                target_offset = np.array(target).transpose(1, 0, 2)  # [npts, nkps, c]
                kp_targ_ofst[msk_idx, :, :] = target_offset[msk_idx, :, :]
        return RTs, kp3ds, ctr3ds, cls_ids, kp_targ_ofst, ctr_targ_ofst

    def __len__(self):
        return len(self.all_lst)

    def __getitem__(self, idx):
        if self.dataset_name == 'train':
            item_name = self.real_syn_gen()
            
            data = self.get_item(item_name)
            while data is None:
                item_name = self.real_syn_gen()
                data = self.get_item(item_name)
            return data
        else:
            item_name = self.all_lst[idx]
            return self.get_item(item_name)

