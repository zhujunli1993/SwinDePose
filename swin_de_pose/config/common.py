#!/usr/bin/env python3
import os
import yaml
import numpy as np
from config.options import BaseOptions

def ensure_fd(fd):
    if not os.path.exists(fd):
        os.system('mkdir -p {}'.format(fd))

opt = BaseOptions().parse()
class ConfigRandLA:
    if opt.crop:
        opt.height, opt.width = opt.max_h, opt.max_w
    k_n = opt.k_n  # KNN
    num_layers = opt.num_layers  # Number of layers
    num_points = opt.height * opt.width // 16  # Number of input points
    num_classes = 22  # Number of valid classes, 22 for YCBV dataset
    sub_grid_size = opt.sub_grid_size  # preprocess_parameter

    batch_size = opt.batch_size  # batch_size during training
    val_batch_size = opt.val_batch_size  # batch_size during validation and test
    train_steps = opt.train_steps  # Number of steps per epochs
    val_steps = opt.val_steps  # Number of validation steps per epoch
    in_c = int(opt.in_c)
    sub_sampling_ratio = opt.sub_sampling_ratio  # sampling ratio of random sampling at each layer
    d_out = opt.d_out  # feature dimension
    num_sub_points = [num_points // 4, num_points // 16, num_points // 64, num_points // 256]

class PspNet:
    psp_out = opt.psp_out
    

class Config:
    def __init__(self, ds_name='ycb', cls_type=''):
    
        self.dataset_name = opt.dataset_name
        self.exp_dir = opt.exp_dir
        self.n_keypoints = opt.n_keypoints
        self.preprocessed_testset_pth = ''
        if self.dataset_name == 'ycb':
            self.ycb_root = opt.data_root
            self.n_objects = 21 + 1  # 21 objects + background
            self.n_classes = self.n_objects
            self.use_orbfps = True
            self.train_path=os.path.join(self.ycb_root, 'dataset_config', opt.train_list)
            self.test_path=os.path.join(self.ycb_root, 'dataset_config', opt.test_list)
            self.kp_orbfps_dir = os.path.join(self.ycb_root, 'ycb_kps')
            self.kp_orbfps_ptn = os.path.join(self.kp_orbfps_dir, '%s_%d_kps.txt')
            self.ycb_cls_lst_p = os.path.join(self.ycb_root, 'dataset_config', 'classes.txt')
            self.ycb_kps_dir = os.path.join(self.ycb_root, 'ycb_kps')
            
            ycb_r_lst_p = os.path.join(self.ycb_root, 'dataset_config', 'radius.txt')
            self.ycb_r_lst = list(np.loadtxt(ycb_r_lst_p))
            self.ycb_cls_lst = self.read_lines(self.ycb_cls_lst_p)
            self.ycb_sym_cls_ids = [13, 16, 19, 20, 21]
        elif self.dataset_name == "linemod":  # linemod
            cls_type = opt.linemod_cls
            self.n_objects = 1 + 1  # 1 object + background
            self.n_classes = self.n_objects
            self.lm_cls_lst = [
                1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15
            ]
            self.lm_sym_cls_ids = [10, 11]
            self.lm_obj_dict = {
                'ape': 1,
                'benchvise': 2,
                'cam': 4,
                'can': 5,
                'cat': 6,
                'driller': 8,
                'duck': 9,
                'eggbox': 10,
                'glue': 11,
                'holepuncher': 12,
                'iron': 13,
                'lamp': 14,
                'phone': 15,
            }
            
            try:
                self.cls_id = self.lm_obj_dict[cls_type]
            except Exception:
                pass
            
            self.lm_id2obj_dict = dict(
                zip(self.lm_obj_dict.values(), self.lm_obj_dict.keys())
            )
            self.lm_root = opt.data_root
            self.cls_root = os.path.join(self.lm_root, "data/%02d/" % self.cls_id)
            self.train_path = os.path.join(self.cls_root, opt.train_list)
            self.render_path = os.path.join(self.lm_root, 'renders/%s/*.pkl' % cls_type)
            self.fuse_path = os.path.join(self.lm_root, 'fuse/%s/*.pkl' % cls_type)
            self.test_path = os.path.join(self.cls_root, opt.test_list)
            if not opt.lm_no_fuse and not opt.lm_no_render:
                self.render_files = os.path.join(self.lm_root, 'renders_nrm/%s/file_list.txt' % cls_type)
                self.fuse_files = os.path.join(self.lm_root, 'fuse_nrm/%s/file_list.txt' % cls_type)
            if not opt.lm_no_pbr:    
                self.pbr_mask_files = os.path.join(self.lm_root, 'train_pbr/train_pbr_obj_%d.txt' % self.cls_id)
            self.use_orbfps = True
            self.kp_orbfps_dir = os.path.join(self.lm_root, 'kps_orb9_fps')
            
            self.kp_orbfps_ptn = os.path.join(self.kp_orbfps_dir, '%s_%d_kps.txt')
            # FPS
            self.lm_fps_kps_dir = os.path.join(self.lm_root, 'lm_obj_kps')

            lm_r_pth = os.path.join(self.lm_root, "dataset_config/models_info.yml")
            lm_r_file = open(os.path.join(lm_r_pth), "r")
            self.lm_r_lst = yaml.safe_load(lm_r_file)

            self.val_nid_ptn = "/data/6D_Pose_Data/datasets/LINEMOD/pose_nori_lists/{}_real_val.nori.list"
        elif self.dataset_name == "lab":

            cls_type = opt.linemod_cls
            self.n_objects = 1 + 1  # 1 object + background
            self.n_classes = self.n_objects
            self.lm_cls_lst = [
                1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15
            ]
            self.lm_sym_cls_ids = [10, 11]
            self.lm_obj_dict = {
                'ape': 1,
                'benchvise': 2,
                'cam': 4,
                'can': 5,
                'cat': 6,
                'driller': 8,
                'duck': 9,
                'eggbox': 10,
                'glue': 11,
                'holepuncher': 12,
                'iron': 13,
                'lamp': 14,
                'phone': 15,
            }
            
            try:
                self.cls_id = self.lm_obj_dict[cls_type]
            except Exception:
                pass
            
            self.lm_id2obj_dict = dict(
                zip(self.lm_obj_dict.values(), self.lm_obj_dict.keys())
            )
            self.lm_root = opt.data_root
            self.depth_input = opt.lab_depth_input
            self.use_orbfps = True
            self.kp_orbfps_dir = os.path.join(self.lm_root, 'kps_orb9_fps')
            self.kp_orbfps_ptn = os.path.join(self.kp_orbfps_dir, '%s_%d_kps.txt')
            # FPS
            self.lm_fps_kps_dir = os.path.join(self.lm_root, 'lm_obj_kps')
            lm_r_pth = os.path.join(self.lm_root, "dataset_config/models_info.yml")
            lm_r_file = open(os.path.join(lm_r_pth), "r")
            self.lm_r_lst = yaml.safe_load(lm_r_file)

        else:  # occlusion linemod
            cls_type = opt.occ_linemod_cls
            self.n_objects = 1 + 1  # 1 object + background
            self.n_classes = self.n_objects
            self.lmo_cls_lst = [
                1, 5, 6, 8, 9, 10, 11, 12
            ]
            self.lm_sym_cls_ids = [10, 11]
            self.lmo_obj_dict = {
                'ape': 1,
                'can': 5,
                'cat': 6,
                'driller': 8,
                'duck': 9,
                'eggbox': 10,
                'glue': 11,
                'holepuncher': 12
            }
            
            try:
                self.cls_id = self.lmo_obj_dict[cls_type]
            except Exception:
                pass
            
            self.lmo_id2obj_dict = dict(
                zip(self.lmo_obj_dict.values(), self.lmo_obj_dict.keys())
            )
            self.occ_lm_root = opt.data_root
            self.train_path = os.path.join(self.occ_lm_root, opt.train_list)
            self.test_path = os.path.join(self.occ_lm_root, opt.test_list)
            # self.cls_root = os.path.join(self.occ_lm_root, "data/%02d/" % self.cls_id)
            # self.train_path = os.path.join(self.cls_root, opt.train_list)
            # self.render_path = os.path.join(self.lm_root, 'renders/%s/*.pkl' % cls_type)
            # self.fuse_path = os.path.join(self.lm_root, 'fuse/%s/*.pkl' % cls_type)
            # self.test_path = os.path.join(self.cls_root, opt.test_list)
            # if not opt.lm_no_fuse and not opt.lm_no_render:
            #     self.render_files = os.path.join(self.lm_root, 'renders_nrm/%s/file_list.txt' % cls_type)
            #     self.fuse_files = os.path.join(self.lm_root, 'fuse_nrm/%s/file_list.txt' % cls_type)
            #     self.render_files_rgb = os.path.join(self.lm_root, 'renders/%s/file_list.txt' % cls_type)
            #     self.fuse_files_rgb = os.path.join(self.lm_root, 'fuse/%s/file_list.txt' % cls_type)
            # if not opt.lm_no_pbr:    
            #     self.pbr_mask_files = os.path.join(self.lm_root, 'train_pbr/train_pbr_obj_%d.txt' % self.cls_id)
            self.use_orbfps = True
            self.kp_orbfps_dir = os.path.join(self.occ_lm_root, 'kps_orb9_fps')
            
            self.kp_orbfps_ptn = os.path.join(self.kp_orbfps_dir, '%s_%d_kps.txt')
            # FPS
            self.lm_fps_kps_dir = os.path.join(self.occ_lm_root, 'lm_obj_kps')

            lm_r_pth = os.path.join(self.occ_lm_root, "models/models_info.yml")
            lm_r_file = open(os.path.join(lm_r_pth), "r")
            self.lm_r_lst = yaml.safe_load(lm_r_file)

            self.val_nid_ptn = "/data/6D_Pose_Data/datasets/LINEMOD/pose_nori_lists/{}_real_val.nori.list"
            
            
        self.intrinsic_matrix = {
            'linemod': np.array([[572.4114, 0.,         325.2611],
                                [0.,        573.57043,  242.04899],
                                [0.,        0.,         1.]]),
            'blender': np.array([[700.,     0.,     320.],
                                 [0.,       700.,   240.],
                                 [0.,       0.,     1.]]),
            'ycb_K1': np.array([[1066.778, 0.        , 312.9869],
                                [0.      , 1067.487  , 241.3109],
                                [0.      , 0.        , 1.0]], np.float32),
            'ycb_K2': np.array([[1077.836, 0.        , 323.7872],
                                [0.      , 1078.189  , 279.6921],
                                [0.      , 0.        , 1.0]], np.float32),
            'lab': np.array([[528.1860, 0, 324.61160],
                     [0, 520.4914, 234.4936],
                     [0, 0, 1]], np.float32)
        }

    def read_lines(self, p):
        with open(p, 'r') as f:
            return [
                line.strip() for line in f.readlines()
            ]


config = Config()
# vim: ts=4 sw=4 sts=4 expandtab
