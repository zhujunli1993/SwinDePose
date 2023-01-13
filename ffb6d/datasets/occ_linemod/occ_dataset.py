#!/usr/bin/env python3
from torchvision import transforms
import torch
import numpy as np
import cv2
import os
import random
from torch.utils.data import Dataset
import torch
import os.path
from PIL import Image
from config.common import Config
from config.options import BaseOptions
import pickle as pkl
from utils.basic_utils import Basic_Utils
import yaml
import scipy.io as scio
import scipy.misc
from glob import glob
import normalSpeed
from models.RandLA.helper_tool import DataProcessing as DP
try:
    from neupeak.utils.webcv2 import imshow, waitKey
except ImportError:
    from cv2 import imshow, waitKey
import math
import pdb

# for get depth_filling function
config_fill = Config(ds_name='ycb')
bs_utils_fill = Basic_Utils(config_fill)
# cuda = torch.cuda.is_available()

class OcclusionLinemodDataset(Dataset):
    def __init__(self,
                 base_dir='/workspace/DATA/OCCLUSION_LINEMOD',
                 object_name='all',
                 dataset_name='test', DEBUG=False):
        self.opt = BaseOptions().parse()
        self.config = Config(ds_name='occlusion_linemod', cls_type=self.opt.occ_linemod_cls)
        self.cls_type = self.opt.occ_linemod_cls
        self.bs_utils = Basic_Utils(self.config)
        self.camera_intrinsic = {'fu': 572.41140, 'fv': 573.57043,
                                 'uc': 325.26110, 'vc': 242.04899}
        self.K = np.matrix([[self.camera_intrinsic['fu'], 0, self.camera_intrinsic['uc']],
                            [0, self.camera_intrinsic['fv'], self.camera_intrinsic['vc']],
                            [0, 0, 1]], dtype=np.float32)
        self.img_shape = (480, 640) # (h, w)
        self.xmap = np.array([[j for i in range(self.opt.width)] for j in range(self.opt.height)])
        self.ymap = np.array([[i for i in range(self.opt.width)] for j in range(self.opt.height)])

        # use alignment_flipping to correct pose labels
        self.alignment_flipping = np.matrix([[1., 0., 0.],
                                             [0., -1., 0.],
                                             [0., 0., -1.]], dtype=np.float32)
        self.base_dir = base_dir
        linemod_objects = ['ape', 'can', 'cat', 'driller',
                           'duck', 'eggbox', 'glue', 'holepuncher']
        if object_name == 'all':
            self.object_names = linemod_objects
        elif object_name in linemod_objects:
            self.object_names = [object_name]
        else:
            raise ValueError('Invalid object name: {}'.format(object_name))
        # compute length
        self.lengths = {}
        self.total_length = 0
        for object_name in self.object_names:
            length = len(list(filter(lambda x: x.endswith('txt'),
                                     os.listdir(os.path.join(base_dir, 'valid_poses', object_name)))))
            self.lengths[object_name] = length
            self.total_length += length
        # pre-load data into memory
        self.pts3d = {}
        self.normals = {}
        self.R_lo = {}
        self.t_lo = {}
    
        for object_name in self.object_names:
            # transformations
            self.R_lo[object_name], self.t_lo[object_name] = \
                    self.get_linemod_to_occlusion_transformation(object_name)
            # keypoints
            # if self.opt.n_keypoints == 8:
            #     kp_type = 'farthest'
            # else:
            #     kp_type = 'farthest{}'.format(self.opt.n_keypoints)
            # pts3d = self.bs_utils.get_kps(
            #     self.cls_type, kp_type=kp_type, ds_type='linemod'
            # )
            # # pts3d_name = os.path.join('data/linemod', 'keypoints',
            # #                           object_name, 'keypoints_3d.npy')
            # # pts3d = np.float32(np.load(pts3d_name))
            # self.pts3d[object_name] = pts3d
            # symmetry plane normals
            # normal_name = os.path.join('data/linemod', 'symmetries',
            #                            object_name, 'symmetries.txt')
            # self.normals[object_name] = self.read_normal(normal_name)
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return self.total_length

    def get_linemod_to_occlusion_transformation(self, object_name):
        # https://github.com/ClayFlannigan/icp
        
        if object_name == 'ape':
            R = np.array([[-4.5991463e-08, -1.0000000e+00,  1.1828890e-08],
                          [ 8.5046146e-08, -4.4907327e-09, -1.0000000e+00],
                          [ 1.0000000e+00, -2.7365417e-09,  9.5073148e-08]], dtype=np.float32)
            t = np.array([ 0.00464956, -0.04454319, -0.00454451], dtype=np.float32)
        elif object_name == 'can':
            R = np.array([[ 1.5503679e-07, -1.0000000e+00,  2.0980373e-07],
                          [ 2.6769550e-08, -2.0030792e-07, -1.0000000e+00],
                          [ 1.0000000e+00,  1.5713613e-07,  2.8610597e-08]], dtype=np.float32)
            t = np.array([-0.009928,   -0.08974387, -0.00697199], dtype=np.float32)
        elif object_name == 'cat':
            R = np.array([[-7.1956642e-08, -1.0000000e+00, -7.8242387e-08],
                          [-9.9875002e-08,  6.7945813e-08, -1.0000000e+00],
                          [ 1.0000000e+00, -6.8791721e-08, -1.0492791e-07]], dtype=np.float32)
            t = np.array([-0.01460595, -0.05390565,  0.00600646], dtype=np.float32)
        elif object_name == 'driller':
            R = np.array([[-5.8952626e-08, -9.9999994e-01,  1.7797127e-07],
                          [ 6.7603776e-09, -1.7821345e-07, -1.0000000e+00],
                          [ 9.9999994e-01, -5.8378635e-08,  2.7301144e-08]], dtype=np.float32)
            t = np.array([-0.00176942, -0.10016585,  0.00840302], dtype=np.float32)
        elif object_name == 'duck':
            R = np.array([[-3.4352450e-07, -1.0000000e+00,  4.5238485e-07],
                          [-6.4654046e-08, -4.5092108e-07, -1.0000000e+00],
                          [ 1.0000000e+00, -3.4280166e-07, -4.6047357e-09]], dtype=np.float32)
            t = np.array([-0.00285449, -0.04044429,  0.00110274], dtype=np.float32)
        elif object_name == 'eggbox':
            R = np.array([[-0.02, -1.00, 0.00],
                          [-0.02, -0.00, -1.00],
                          [1.00, -0.02, -0.02]], dtype=np.float32)
            t = np.array([-0.01, -0.03, -0.00], dtype=np.float32)
        elif object_name == 'glue':
            R = np.array([[-1.2898508e-07, -1.0000000e+00,  6.7859062e-08],
                          [ 2.9789486e-08, -6.8855734e-08, -9.9999994e-01],
                          [ 1.0000000e+00, -1.2711939e-07,  2.9696672e-08]], dtype=np.float32)
            t = np.array([-0.00144855, -0.07744411, -0.00468425], dtype=np.float32)
        elif object_name == 'holepuncher':
            R = np.array([[-5.9812328e-07, -9.9999994e-01,  3.9026276e-07],
                          [ 8.9670505e-07, -3.8723923e-07, -1.0000001e+00],
                          [ 1.0000000e+00, -5.9914004e-07,  8.8171902e-07]], dtype=np.float32)
            t = np.array([-0.00425799, -0.03734197,  0.00175619], dtype=np.float32)
        t = t.reshape((3, 1))
        return R, t

    def read_pose_and_img_id(self, filename, example_id):
        read_rotation = False
        read_translation = False
        R = []
        T = []
        with open(filename) as f:
            for line in f:
                if read_rotation:
                    R.append(line.split())
                    if len(R) == 3:
                        read_rotation = False
                elif read_translation:
                    T = line.split()
                    read_translation = False
                if line.startswith('rotation'):
                    read_rotation = True
                elif line.startswith('center'):
                    read_translation = True
        
        R = np.array(R, dtype=np.float32) # 3*3
        T = np.array(T, dtype=np.float32).reshape((3, 1)) # 3*1
        img_id = int(line) # in the last line
        return R, T, img_id

    def read_normal(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            normal = np.array(lines[3].strip().split(), dtype=np.float32)
        return normal

    def keypoints_to_map(self, mask, pts2d, unit_vectors=True):
        # based on: https://github.com/zju3dv/pvnet/blob/master/lib/datasets/linemod_dataset.py
        mask = mask[0]
        h, w = mask.shape
        n_pts = pts2d.shape[0]
        xy = np.argwhere(mask == 1.)[:, [1, 0]]
        xy = np.expand_dims(xy.transpose(0, 1), axis=1)
        pts_map = np.tile(xy, (1, n_pts, 1))
        pts_map = np.tile(np.expand_dims(pts2d, axis=0), (pts_map.shape[0], 1, 1)) - pts_map
        if unit_vectors:
            norm = np.linalg.norm(pts_map, axis=2, keepdims=True)
            norm[norm < 1e-3] += 1e-3
            pts_map = pts_map / norm
        pts_map_out = np.zeros((h, w, n_pts, 2), np.float32)
        pts_map_out[xy[:, 0, 1], xy[:, 0, 0]] = pts_map
        pts_map_out = np.reshape(pts_map_out, (h, w, n_pts * 2))
        pts_map_out = np.transpose(pts_map_out, (2, 0, 1))
        return pts_map_out

    def keypoints_to_graph(self, mask, pts2d):
        mask = mask[0]
        num_pts = pts2d.shape[0]
        num_edges = num_pts * (num_pts - 1) // 2
        graph = np.zeros((num_edges, 2, self.img_shape[0], self.img_shape[1]),
                         dtype=np.float32)
        edge_idx = 0
        for start_idx in range(0, num_pts - 1):
            start = pts2d[start_idx]
            for end_idx in range(start_idx + 1, num_pts):
                end = pts2d[end_idx]
                edge = end - start
                graph[edge_idx, 0][mask == 1.] = edge[0]
                graph[edge_idx, 1][mask == 1.] = edge[1]
                edge_idx += 1
        graph = graph.reshape((num_edges * 2, self.img_shape[0], self.img_shape[1]))
        return graph

    def read_3d_points_linemod(self, object_name):
        filename = 'data/linemod/original_dataset/{}/mesh.ply'.format(object_name)
        with open(filename) as f:
            in_vertex_list = False
            vertices = []
            in_mm = False
            for line in f:
                if in_vertex_list:
                    vertex = line.split()[:3]
                    vertex = np.array([float(vertex[0]),
                                       float(vertex[1]),
                                       float(vertex[2])], dtype=np.float32)
                    if in_mm:
                        vertex = vertex / np.float32(10) # mm -> cm
                    vertex = vertex / np.float32(100)
                    vertices.append(vertex)
                    if len(vertices) >= vertex_count:
                        break
                elif line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif line.startswith('end_header'):
                    in_vertex_list = True
                elif line.startswith('element face'):
                    in_mm = True
        return np.matrix(vertices)

    def read_3d_points_occlusion(self, object_name):
        filename = glob.glob('data/occlusion_linemod/models/{}/*.xyz'.format(object_name))[0]
        with open(filename) as f:
            vertices = []
            for line in f:
                vertex = line.split()[:3]
                vertex = np.array([float(vertex[0]),
                                   float(vertex[1]),
                                   float(vertex[2])], dtype=np.float32)
                vertices.append(vertex)
        vertices = np.matrix(vertices)
        return vertices
    
    
            
    def get_pose_gt_info(self, cld, labels, RT):
        RTs = np.zeros((self.config.n_objects, 3, 4))
        kp3ds = np.zeros((self.config.n_objects, self.opt.n_keypoints, 3))
        ctr3ds = np.zeros((self.config.n_objects, 3))
        cls_ids = np.zeros((self.config.n_objects, 1))
        kp_targ_ofst = np.zeros((self.opt.n_sample_points, self.opt.n_keypoints, 3))
        ctr_targ_ofst = np.zeros((self.opt.n_sample_points, 3))
        for i, cls_id in enumerate([1]):
            RTs[i] = RT
            r = RT[:, :3]
            t = RT[:, 3]

            ctr = self.bs_utils.get_ctr(self.cls_type, ds_type="linemod")[:, None]
            ctr = np.dot(ctr.T, r.T) + t
            ctr3ds[i, :] = ctr[0]
            msk_idx = np.where(labels == cls_id)[0]

            target_offset = np.array(np.add(cld, -1.0*ctr3ds[i, :]))
            ctr_targ_ofst[msk_idx, :] = target_offset[msk_idx, :]
            cls_ids[i, :] = np.array([1])

            if self.opt.n_keypoints == 8:
                kp_type = 'farthest'
            else:
                kp_type = 'farthest{}'.format(self.opt.n_keypoints)
            kps = self.bs_utils.get_kps(
                self.cls_type, kp_type=kp_type, ds_type='linemod'
            )
            # pts3d_name = os.path.join('data/linemod', 'keypoints',
            #                           object_name, 'keypoints_3d.npy')
            # pts3d = np.float32(np.load(pts3d_name))
            
            kps = np.dot(kps, r.T) + t
            kp3ds[i] = kps
            ############################
            # pts3d = self.pts3d[object_name]
            # pts3d_trans = np.matrix(R) * np.matrix(pts3d).transpose() + np.matrix(t)
            # pts3d_trans = np.array(pts3d_trans)
            ##########################
            target = []
            for kp in kps:
                target.append(np.add(cld, -1.0*kp))
            target_offset = np.array(target).transpose(1, 0, 2)  # [npts, nkps, c]
            kp_targ_ofst[msk_idx, :, :] = target_offset[msk_idx, :, :]

        return RTs, kp3ds, ctr3ds, cls_ids, kp_targ_ofst, ctr_targ_ofst
    def scale_pseudo(self, pseudo):
        
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
    
    
    
    def dpt_2_pcld(self, dpt, cam_scale, K):
        if len(dpt.shape) > 2:
            dpt = dpt[:, :, 0]
        dpt = dpt.astype(np.float32) / cam_scale
        msk = (dpt > 1e-8).astype(np.float32)
        row = (self.ymap - K[0,2]) * dpt / K[0,0]
        col = (self.xmap - K[1,2]) * dpt / K[1,1]
        dpt_3d = np.concatenate(
            (row[..., None], col[..., None], dpt[..., None]), axis=2
        )
        dpt_3d = dpt_3d * msk[:, :, None]
        return dpt_3d
    
    def __getitem__(self, idx):
        
        local_idx = idx
        for object_name in self.object_names:
            if local_idx < self.lengths[object_name]:
                # pose
                pose_name = os.path.join(self.base_dir, 'valid_poses', object_name, '{}.txt'.format(local_idx))
                R, t, img_id = self.read_pose_and_img_id(pose_name, local_idx)
                R = np.array(self.alignment_flipping * R, dtype=np.float32)
                t = np.array(self.alignment_flipping * t, dtype=np.float32)
                # apply linemod->occlusion alignment
                t = np.matmul(R, self.t_lo[object_name]) + t
                R = np.matmul(R, self.R_lo[object_name])
                RT = np.concatenate((R, t), axis=1)
                # image
                with np.load(os.path.join(self.base_dir, 'data', object_name, "pseudo_nrm_angles/{:04d}.npz".format(img_id))) as data:
                    angles = data['angles']
                # convert angles and signed angles to image range (0~255)
                pdb.set_trace()
                sed_angles = self.scale_pseudo(angles)
                sed_angles = Image.fromarray(np.uint8(sed_angles))
                nrm_angles = np.array(sed_angles)[:, :, :3]
                # mask
                with Image.open(os.path.join(self.base_dir, 'masks', object_name, '{}.png'.format(img_id))) as li:
                    labels = np.array(li)
                    labels = (labels > 0).astype("uint8")
                # depth
                with Image.open(os.path.join(self.base_dir, 'RGB-D', 'depth_noseg', 'depth_{:05d}.png'.format(img_id))) as di:
                    dpt_mm = np.array(di)
                cam_scale = 1000.0
                dpt_mm = bs_utils_fill.fill_missing(dpt_mm, cam_scale, 1)
                dpt_mm = dpt_mm.copy().astype(np.uint16)
                nrm_map = normalSpeed.depth_normal(
                    dpt_mm, self.K[0,0], self.K[1,1], 5, 2000, 20, False
                )
                # if self.DEBUG:
                #     show_nrm_map = ((nrm_map + 1.0) * 127).astype(np.uint8)
                #     imshow("nrm_map", show_nrm_map)

                dpt_m = dpt_mm.astype(np.float32) / cam_scale
                dpt_xyz = self.dpt_2_pcld(dpt_m, 1.0, self.K)
                dpt_xyz[np.isnan(dpt_xyz)] = 0.0
                dpt_xyz[np.isinf(dpt_xyz)] = 0.0
                if len(labels.shape) > 2:
                    labels = labels[:, :, 0]
                rgb_labels = labels.copy()
                
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
                labels_pt = labels.flatten()[choose]
                choose = np.array([choose])
                cld_angle_nrm = np.concatenate((cld, nrm_angles_pt, nrm_pt), axis=1).transpose(1, 0)        
                
                # keypoints
                RTs, kp3ds, ctr3ds, cls_ids, kp_targ_ofst, ctr_targ_ofst = self.get_pose_gt_info(cld, labels_pt, RT)
                
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
                
                sr2dptxyz = {
                    pow(2, ii): item.reshape(3, -1).transpose(1, 0)
                    for ii, item in enumerate(xyz_lst)
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
                    inputs['cld_xyz%d' % i] = cld.astype(np.float32).copy()
                    inputs['cld_nei_idx%d' % i] = nei_idx.astype(np.int32).copy()
                    
                    inputs['cld_sub_idx%d' % i] = pool_i.astype(np.int32).copy()
                    inputs['cld_interp_idx%d' % i] = up_i.astype(np.int32).copy()
                    # nei_r2p = DP.knn_search(
                    #     sr2dptxyz[rgb_ds_sr[i]][None, ...], sub_pts[None, ...], 16
                    # ).astype(np.int32).squeeze(0)
                    #inputs['r2p_ds_nei_idx%d' % i] = nei_r2p.copy()
                    
                    # nei_p2r = DP.knn_search(
                    #     sub_pts[None, ...], sr2dptxyz[rgb_ds_sr[i]][None, ...], 1
                    # ).astype(np.int32).squeeze(0)
                    #inputs['p2r_ds_nei_idx%d' % i] = nei_p2r.copy()
                    cld = sub_pts

                n_up_layers = 3
                rgb_up_sr = [4, 2, 2]
                for i in range(n_up_layers):
                    r2p_nei = DP.knn_search(
                        sr2dptxyz[rgb_up_sr[i]][None, ...],
                        inputs['cld_xyz%d'%(n_ds_layers-i-1)][None, ...], 16
                    ).astype(np.int32).squeeze(0)
                    #inputs['r2p_up_nei_idx%d' % i] = r2p_nei.copy()
                    p2r_nei = DP.knn_search(
                        inputs['cld_xyz%d'%(n_ds_layers-i-1)][None, ...],
                        sr2dptxyz[rgb_up_sr[i]][None, ...], 1
                    ).astype(np.int32).squeeze(0)
                    #inputs['p2r_up_nei_idx%d' % i] = p2r_nei.copy()

                # show_rgb = rgb.transpose(1, 2, 0).copy()[:, :, ::-1]
                # if self.DEBUG:
                #     for ip, xyz in enumerate(xyz_lst):
                #         pcld = xyz.reshape(3, -1).transpose(1, 0)
                #         p2ds = self.bs_utils.project_p3d(pcld, cam_scale, K)
                #         srgb = self.bs_utils.paste_p2ds(show_rgb.copy(), p2ds, (0, 0, 255))
                        # imshow("rz_pcld_%d" % ip, srgb)
                        # p2ds = self.bs_utils.project_p3d(inputs['cld_xyz%d'%ip], cam_scale, K)
                        # srgb1 = self.bs_utils.paste_p2ds(show_rgb.copy(), p2ds, (0, 0, 255))
                        # imshow("rz_pcld_%d_rnd" % ip, srgb1)
                # print(
                #     "kp3ds:", kp3ds.shape, kp3ds, "\n",
                #     "kp3ds.mean:", np.mean(kp3ds, axis=0), "\n",
                #     "ctr3ds:", ctr3ds.shape, ctr3ds, "\n",
                #     "cls_ids:", cls_ids, "\n",
                #     "labels.unique:", np.unique(labels),
                # )
                # if ".npz" in item_name:
                #     item_name = item_name.split('/')[-1].split('.')[0]
                item_dict = dict(
                    img_id=np.uint8(img_id),
                    nrm_angles=nrm_angles.astype(np.uint8),  # [c, h, w]
                    cld_angle_nrm=cld_angle_nrm.astype(np.float32),  # [9, npts]
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
        # if self.DEBUG:
        #     extra_d = dict(
        #         dpt_xyz_nrm=dpt_6c.astype(np.float32),  # [6, h, w]
        #         cam_scale=np.array([cam_scale]).astype(np.float32),
        #         K=K.astype(np.float32),
        #     )
        #     item_dict.update(extra_d)
        #     item_dict['normal_map'] = nrm_map[:, :, :3].astype(np.float32)
        return item_dict


if __name__ == '__main__':
    linemod_objects = ['ape', 'can', 'cat', 'driller',
                       'duck', 'eggbox', 'glue', 'holepuncher']
    for name in linemod_objects:
        dataset = OcclusionLinemodDataset(object_name=name)
        print(name, len(dataset))