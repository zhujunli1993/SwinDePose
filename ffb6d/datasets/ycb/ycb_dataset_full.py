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
        
        self.debug = DEBUG
        self.xmap = np.array([[j for i in range(opt.width)] for j in range(opt.height)])
        self.ymap = np.array([[i for i in range(opt.width)] for j in range(opt.height)])
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

    def real_syn_gen(self):
        if self.rng.rand() > 0.8:
            n = len(self.real_lst)
            idx = self.rng.randint(0, n)
            item = self.real_lst[idx]
        else:
            n = len(self.syn_lst)
            idx = self.rng.randint(0, n)
            item = self.syn_lst[idx]
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
        # just for debug
        
        
        with Image.open(os.path.join(self.root, item_name+'-depth.png')) as di:
            dpt_um = np.array(di)
        with Image.open(os.path.join(self.root, item_name+'-label.png')) as li:
            labels = np.array(li)
        rgb_labels = labels.copy()
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
            
        rnd_typ = 'syn' if 'syn' in item_name else 'real'
        cam_scale = meta['factor_depth'].astype(np.float32)[0][0]
        msk_dp = dpt_um > 1e-6

        if self.add_noise and rnd_typ == 'syn':
            rgb = self.rgb_add_noise(rgb)
            rgb_s = self.rgb_add_noise(rgb_s)
            # Save initial noised images
            # img = np.uint8((rgb * 0.5 + 0.5)[:, :, ::-1] * 255.0)
            # img_s = np.uint8((rgb_s * 0.5 + 0.5)[:, :, ::-1] * 255.0)
            # cv2.imwrite('/workspace/REPO/pose_estimation/check_train_rgb_1.png', img)
            # cv2.imwrite('/workspace/REPO/pose_estimation/check_train_rgb_s_1.png', img_s)
            
            # Add background to initially noised images
            rgb, rgb_s, dpt_um = self.add_real_back(rgb, rgb_s, rgb_labels, dpt_um, msk_dp)
            
            # Save noised images with background 
            # img = np.uint8((rgb * 0.5 + 0.5)[:, :, ::-1] * 255.0)
            # img_s = np.uint8((rgb_s * 0.5 + 0.5)[:, :, ::-1] * 255.0)
            # cv2.imwrite('/workspace/REPO/pose_estimation/check_train_rgb_2.png', img)
            # cv2.imwrite('/workspace/REPO/pose_estimation/check_train_rgb_s_2.png', img_s)
            
            # Force adding additional noise
            if self.rng.rand() > 0.8:
                rgb = self.rgb_add_noise(rgb)
                rgb_s = self.rgb_add_noise(rgb_s)
                # Save additional noised images with background 
                # img = np.uint8((rgb * 0.5 + 0.5)[:, :, ::-1] * 255.0)
                # img_s = np.uint8((rgb_s * 0.5 + 0.5)[:, :, ::-1] * 255.0)
                # cv2.imwrite('/workspace/REPO/pose_estimation/check_train_rgb_3.png', img)
                # cv2.imwrite('/workspace/REPO/pose_estimation/check_train_rgb_s_3.png', img_s)
        # testing the pts in image

        # rot = render_data['calib'][i,:3, :3]
        # trans = render_data['calib'][i,:3, 3:4]
        # pts = torch.addmm(trans, rot, sample_data['samples'][:, sample_data['labels'][0] > 0.5])  # [3, N]
        # pts = torch.zeros([1, 3])
        # offset = torch.tensor([0, 0, 0])
        # pts = torch.add(pts, offset)
        # pts = torch.addmm(trans, rot, pts.T)  # [3, N]
        # pts = 0.5 * (pts.numpy().T + 1.0) * render_data['img'].size(2)
        # p = torch.tensor([0, 0, 0])
        # p = torch.addmm(trans, rot, p)  # [3, N]
        # p = 0.5 * (p.numpy().T + 1.0) * render_data['img'].size(2)
        # for p in pts:
        #     img = cv2.circle(cv2.UMat(img), (p[0], p[1]), 2, (0,255,0), -1)
        # print('Original Mesh Center(0,0,0) in image '+str(i)+': ', p)

        
        rgb_c = np.concatenate((rgb, rgb_s), axis=2)  #[h,w,6]
        dpt_um = bs_utils.fill_missing(dpt_um, cam_scale, 1)
        msk_dp = dpt_um > 1e-6

        dpt_mm = (dpt_um.copy()/10).astype(np.uint16)
        nrm_map = normalSpeed.depth_normal(
            dpt_mm, K[0][0], K[1][1], 5, 2000, 20, False
        )
        # save normal map
        # if True:
        #     show_nrm_map = ((nrm_map + 1.0) * 127).astype(np.uint8)
        #     cv2.imwrite('/workspace/REPO/pose_estimation/nrm_map.png', show_nrm_map)
        
        
        dpt_m = dpt_um.astype(np.float32) / cam_scale
        dpt_xyz = self.dpt_2_pcld(dpt_m, 1.0, K)

        choose = msk_dp.flatten().nonzero()[0].astype(np.uint32)
        if len(choose) < 400:
            return None
        choose_2 = np.array([i for i in range(len(choose))])
        if len(choose_2) < 400:
            return None
        if len(choose_2) > opt.n_sample_points:
            c_mask = np.zeros(len(choose_2), dtype=int)
            c_mask[:opt.n_sample_points] = 1
            np.random.shuffle(c_mask)
            choose_2 = choose_2[c_mask.nonzero()]
        else:
            choose_2 = np.pad(choose_2, (0, opt.n_sample_points-len(choose_2)), 'wrap')
        choose = np.array(choose)[choose_2]

        sf_idx = np.arange(choose.shape[0])
        np.random.shuffle(sf_idx)
        choose = choose[sf_idx]

        cld = dpt_xyz.reshape(-1, 3)[choose, :]
        rgb_c_pt = rgb_c.reshape(-1, 6)[choose, :].astype(np.float32)
        
        nrm_pt = nrm_map[:, :, :3].reshape(-1, 3)[choose, :]
        labels_pt = labels.flatten()[choose]
        choose = np.array([choose])
        cld_rgb_nrm = np.concatenate((cld, rgb_c_pt, nrm_pt), axis=1).transpose(1, 0)
        
        # Save points with normal
        # save_file = np.zeros([6, choose.shape[1]])
        # save_file[0:3,:] = cld.transpose(1,0)
        # save_file[3:, :] = nrm_pt.transpose(1,0)
        # np.savetxt('/workspace/REPO/pose_estimation/c_normal.txt',
        #             save_file.T,
        #             fmt='%.6f %.6f %.6f %.6f %.6f %.6f',
        #             comments=''
        #             )
        
        cls_id_lst = meta['cls_indexes'].flatten().astype(np.uint32)
        
        RTs, kp3ds, ctr3ds, cls_ids, kp_targ_ofst, ctr_targ_ofst = self.get_pose_gt_info(
            cld, labels_pt, cls_id_lst, meta
        )

        h, w = opt.height, opt.width
        rgb_c = np.transpose(rgb_c, (2, 0, 1)) # hwc2chw
        
        xyz_lst = [dpt_xyz.transpose(2, 0, 1)] # c, h, w
        
        # Set downsampling index
        for i in range(3):
            scale = pow(2, i+1)
            nh, nw = h // pow(2, i+1), w // pow(2, i+1)
            ys, xs = np.mgrid[:nh, :nw]
            xyz_lst.append(xyz_lst[0][:, ys*scale, xs*scale])
            
        
        # Save points with normal
        # save_file = np.zeros([3, choose.shape[1]])
        # save_file[0:3,:] = cld.transpose(1,0)
        # save_file[3:, :] = nrm_pt.transpose(1,0)
        # np.savetxt('/workspace/REPO/pose_estimation/c_normal.txt',
        #             save_file.T,
        #             fmt='%.6f %.6f %.6f %.6f %.6f %.6f',
        #             comments=''
        #             )
        
        # Save each stage of downsampling into a dict
        sr2dptxyz = {
            pow(2, ii): item.reshape(3, -1).transpose(1, 0) for ii, item in enumerate(xyz_lst)
        }

        
        rgb_ds_sr = [4, 8, 8, 8]
        n_ds_layers = 4
        pcld_sub_s_r = [4, 4, 4, 4]
        inputs = {}
        
        # DownSample stage
        for i in range(n_ds_layers):
            # Obtain each point's 16 neighbors
            nei_idx = DP.knn_search(
                cld[None, ...], cld[None, ...], 16
            ).astype(np.int32).squeeze(0)
            # Get subset of points 
            sub_pts = cld[:cld.shape[0] // pcld_sub_s_r[i], :]
            # Get neighbors of sub_pts
            pool_i = nei_idx[:cld.shape[0] // pcld_sub_s_r[i], :]
            # Obtain each point's 1 neighbors from sub_pts
            up_i = DP.knn_search(
                sub_pts[None, ...], cld[None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs['cld_xyz%d'%i] = cld.astype(np.float32).copy()
            inputs['cld_nei_idx%d'%i] = nei_idx.astype(np.int32).copy()
            inputs['cld_sub_idx%d'%i] = pool_i.astype(np.int32).copy()
            inputs['cld_interp_idx%d'%i] = up_i.astype(np.int32).copy()
            # Obtain 16 neighbors of each sub_pts point from third and fourth level of original xyz converted from depth image.
            nei_r2p = DP.knn_search(
                sr2dptxyz[rgb_ds_sr[i]][None, ...], sub_pts[None, ...], 16
            ).astype(np.int32).squeeze(0)
            inputs['r2p_ds_nei_idx%d'%i] = nei_r2p.copy()
            
            
            # save nei_r2p points
            # save_file = np.zeros([sr2dptxyz[rgb_ds_sr[i]].shape[0], 6])
            # save_file[:, 0:3] = sr2dptxyz[rgb_ds_sr[i]][None, ...]
            # save_file[:, 3:] = np.full((sr2dptxyz[rgb_ds_sr[i]].shape[0],3), [24,225,77])
            # np.savetxt('/workspace/REPO/pose_estimation/sr2dptxyz_%d_%d.txt'%(rgb_ds_sr[i],i),
            #             save_file,
            #             fmt='%.6f %.6f %.6f %.6f %.6f %.6f',
            #             comments=''
            #             )

            # Obtain 1 neighbors of each third and fourth level of original xyz points, converted from depth image, from each sub_pts point.
            nei_p2r = DP.knn_search(
                sub_pts[None, ...], sr2dptxyz[rgb_ds_sr[i]][None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs['p2r_ds_nei_idx%d'%i] = nei_p2r.copy()
            
            # Update cld to sub_pts, like downsampling. 
            cld = sub_pts
            
            # save cld
            # save_file = np.zeros([cld.shape[0], 3])
            # save_file = cld
            # np.savetxt('/workspace/REPO/pose_estimation/cld_%d.txt'%i,
            #             save_file,
            #             fmt='%.6f %.6f %.6f',
            #             comments=''
            #             )
        
        n_up_layers = 3
        rgb_up_sr = [4, 2, 2]
        for i in range(n_up_layers):
            # Obtain 16 neighbors of each cld point from each downsampling stage of original xyz.
            r2p_nei = DP.knn_search(
                sr2dptxyz[rgb_up_sr[i]][None, ...],
                inputs['cld_xyz%d'%(n_ds_layers-i-1)][None, ...], 16
            ).astype(np.int32).squeeze(0)
            inputs['r2p_up_nei_idx%d'%i] = r2p_nei.copy()
            
            
            # save r2p_nei points
            # save_file = np.zeros([sr2dptxyz[rgb_up_sr[i]].shape[0], 6])
            # save_file[:, 0:3] = sr2dptxyz[rgb_up_sr[i]][None, ...]
            # save_file[:, 3:] = np.full((sr2dptxyz[rgb_up_sr[i]].shape[0],3), [24,225,77])
            # np.savetxt('/workspace/REPO/pose_estimation/sr2dptxyz_up_%d_%d.txt'%(rgb_up_sr[i],i),
            #             save_file,
            #             fmt='%.6f %.6f %.6f %.6f %.6f %.6f',
            #             comments=''
            #             )
            # Obtain 1 neighbors of each downsampling stage of original xyz from cld points
            p2r_nei = DP.knn_search(
                inputs['cld_xyz%d'%(n_ds_layers-i-1)][None, ...],
                sr2dptxyz[rgb_up_sr[i]][None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs['p2r_up_nei_idx%d'%i] = p2r_nei.copy()
        
        
        # show_rgb = rgb.copy()[:, :, ::-1]
        # if True:
        #     for ip, xyz in enumerate(xyz_lst):
        #         pcld = xyz.reshape(3, -1).transpose(1, 0)
        #         p2ds = bs_utils.project_p3d(pcld, cam_scale, K)
        #         print(show_rgb.shape, pcld.shape)
        #         srgb = bs_utils.paste_p2ds(show_rgb.copy(), p2ds, (0, 0, 255))
        #         cv2.imwrite('/workspace/REPO/pose_estimation/rz_pcld_%d.png'% ip, srgb)
        #         p2ds = bs_utils.project_p3d(inputs['cld_xyz%d'%ip], cam_scale, K)
        #         srgb1 = bs_utils.paste_p2ds(show_rgb.copy(), p2ds, (0, 0, 255))
        #         cv2.imwrite('/workspace/REPO/pose_estimation/rz_pcld_%d_d_rnd.png'% ip, srgb1)
        

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
        # if self.debug:
        #     extra_d = dict(
        #         dpt_xyz_nrm=dpt_6c.astype(np.float32),  # [6, h, w]
        #         cam_scale=np.array([cam_scale]).astype(np.float32),
        #         K=K.astype(np.float32),
        #     )
        #     item_dict.update(extra_d)
        #     item_dict['normal_map'] = nrm_map[:, :, :3].astype(np.float32)
        return item_dict

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


