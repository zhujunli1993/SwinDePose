#!/usr/bin/env python3
import os
import cv2
import random
import torch
import numpy as np
from utils.icp import icp
from plyfile import PlyData
import normalSpeed

from utils.ip_basic.ip_basic import vis_utils
from utils.ip_basic.ip_basic import depth_map_utils_ycb as depth_map_utils


intrinsic_matrix = {
    'linemod': np.array([[572.4114, 0.,         325.2611],
                        [0.,        573.57043,  242.04899],
                        [0.,        0.,         1.]]),
    'blender': np.array([[700.,     0.,     320.],
                         [0.,       700.,   240.],
                         [0.,       0.,     1.]]),
    'pascal': np.asarray([[-3000.0, 0.0,    0.0],
                         [0.0,      3000.0, 0.0],
                         [0.0,      0.0,    1.0]]),
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


def VOCap(rec, prec):
    idx = np.where(rec != np.inf)
    if len(idx[0]) == 0:
        return 0
    rec = rec[idx]
    prec = prec[idx]
    mrec = np.array([0.0]+list(rec)+[0.1])
    mpre = np.array([0.0]+list(prec)+[prec[-1]])
    for i in range(1, prec.shape[0]):
        mpre[i] = max(mpre[i], mpre[i-1])
    i = np.where(mrec[1:] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i-1]) * mpre[i]) * 10
    return ap


def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
        A: Nxm numpy array of corresponding points, usually points on mdl
        B: Nxm numpy array of corresponding points, usually points on camera axis
    Returns:
    T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
    R: mxm rotation matrix
    t: mx1 translation vector
    '''

    assert A.shape == B.shape
    # get number of dimensions
    m = A.shape[1]
    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    # rotation matirx
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)
    T = np.zeros((3, 4))
    T[:, :3] = R
    T[:, 3] = t
    return T


class PoseTransformer(object):
    rotation_transform = np.array([[1., 0., 0.],
                                   [0., -1., 0.],
                                   [0., 0., -1.]])
    translation_transforms = {}
    class_type_to_number = {
        'ape': '001',
        'can': '004',
        'cat': '005',
        'driller': '006',
        'duck': '007',
        'eggbox': '008',
        'glue': '009',
        'holepuncher': '010'
    }
    blender_models = {}

    def __init__(self, class_type):
        self.class_type = class_type
        lm_pth = 'datasets/linemod/LINEMOD'
        lm_occ_pth = 'datasets/linemod/OCCLUSION_LINEMOD'
        self.blender_model_path = os.path.join(lm_pth, '{}/{}.ply'.format(class_type, class_type))
        self.xyz_pattern = os.path.join(lm_occ_pth, 'models/{}/{}.xyz')

    @staticmethod
    def load_ply_model(model_path):
        ply = PlyData.read(model_path)
        data = ply.elements[0].data
        x = data['x']
        y = data['y']
        z = data['z']
        return np.stack([x, y, z], axis=-1)

    def get_blender_model(self):
        if self.class_type in self.blender_models:
            return self.blender_models[self.class_type]

        blender_model = self.load_ply_model(self.blender_model_path.format(self.class_type, self.class_type))
        self.blender_models[self.class_type] = blender_model

        return blender_model

    def get_translation_transform(self):
        if self.class_type in self.translation_transforms:
            return self.translation_transforms[self.class_type]

        model = self.get_blender_model()
        xyz = np.loadtxt(self.xyz_pattern.format(
            self.class_type.title(), self.class_type_to_number[self.class_type]))
        rotation = np.array([[0., 0., 1.],
                             [1., 0., 0.],
                             [0., 1., 0.]])
        xyz = np.dot(xyz, rotation.T)
        translation_transform = np.mean(xyz, axis=0) - np.mean(model, axis=0)
        self.translation_transforms[self.class_type] = translation_transform

        return translation_transform

    def occlusion_pose_to_blender_pose(self, pose):
        rot, tra = pose[:, :3], pose[:, 3]
        rotation = np.array([[0., 1., 0.],
                             [0., 0., 1.],
                             [1., 0., 0.]])
        rot = np.dot(rot, rotation)

        tra[1:] *= -1
        translation_transform = np.dot(rot, self.get_translation_transform())
        rot[1:] *= -1
        translation_transform[1:] *= -1
        tra += translation_transform
        pose = np.concatenate([rot, np.reshape(tra, newshape=[3, 1])], axis=-1)

        return pose


class Basic_Utils():

    def __init__(self, config):
        
        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        self.config = config
        if self.config.dataset_name == "ycb":
            self.ycb_cls_lst = config.ycb_cls_lst
        self.ycb_cls_ptsxyz_dict = {}
        self.ycb_cls_ptsxyz_cuda_dict = {}
        self.ycb_cls_kps_dict = {}
        self.ycb_cls_ctr_dict = {}
        self.lm_cls_ptsxyz_dict = {}
        self.lm_cls_ptsxyz_cuda_dict = {}
        self.lm_cls_kps_dict = {}
        self.lm_cls_ctr_dict = {}
        self.lmo_cls_kps_dict = {}
        self.lmo_cls_ctr_dict = {}

    def read_lines(self, p):
        with open(p, 'r') as f:
            lines = [
                line.strip() for line in f.readlines()
            ]
        return lines

    def sv_lines(self, p, line_lst):
        with open(p, 'w') as f:
            for line in line_lst:
                print(line, file=f)

    def translate(self, img, x, y):
        M = np.float32([[1, 0, x], [0, 1, y]])
        shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        return shifted

    def rotate(self, img, angle, ctr=None, scale=1.0):
        (h, w) = img.shape[:2]
        if ctr is None:
            ctr = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(ctr, -1.0 * angle, scale)
        rotated = cv2.warpAffine(img, M, (w, h))
        return rotated

    def cal_degree_from_vec(self, v1, v2):
        cos = np.dot(v1, v2.T) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        if abs(cos) > 1.0:
            cos = 1.0 * (-1.0 if cos < 0 else 1.0)
            print(cos, v1, v2)
        dg = np.arccos(cos) / np.pi * 180
        return dg

    def cal_directional_degree_from_vec(self, v1, v2):
        dg12 = self.cal_degree_from_vec(v1, v2)
        cross = v1[0] * v2[1] - v2[0] * v1[1]
        if cross < 0:
            dg12 = 360 - dg12
        return dg12

    def mean_shift(self, data, radius=5.0):
        clusters = []
        for i in range(len(data)):
            cluster_centroid = data[i]
            cluster_frequency = np.zeros(len(data))
            # Search points in circle
            while True:
                temp_data = []
                for j in range(len(data)):
                    v = data[j]
                    # Handle points in the circles
                    if np.linalg.norm(v - cluster_centroid) <= radius:
                        temp_data.append(v)
                        cluster_frequency[i] += 1
                # Update centroid
                old_centroid = cluster_centroid
                new_centroid = np.average(temp_data, axis=0)
                cluster_centroid = new_centroid
                # Find the mode
                if np.array_equal(new_centroid, old_centroid):
                    break
            # Combined 'same' clusters
            has_same_cluster = False
            for cluster in clusters:
                if np.linalg.norm(cluster['centroid'] - cluster_centroid) <= radius:
                    has_same_cluster = True
                    cluster['frequency'] = cluster['frequency'] + cluster_frequency
                    break
            if not has_same_cluster:
                clusters.append({
                    'centroid': cluster_centroid,
                    'frequency': cluster_frequency
                })

        print('clusters (', len(clusters), '): ', clusters)
        self.clustering(data, clusters)
        return clusters

    # Clustering data using frequency
    def clustering(self, data, clusters):
        t = []
        for cluster in clusters:
            cluster['data'] = []
            t.append(cluster['frequency'])
        t = np.array(t)
        # Clustering
        for i in range(len(data)):
            column_frequency = t[:, i]
            cluster_index = np.where(column_frequency == np.max(column_frequency))[0][0]
            clusters[cluster_index]['data'].append(data[i])

    def project_p3d(self, p3d, cam_scale, K=intrinsic_matrix['ycb_K1']):
        if type(K) == str:
            K = intrinsic_matrix[K]
        p3d = p3d * cam_scale
        p2d = np.dot(p3d, K.T)
        p2d_3 = p2d[:, 2]
        p2d_3[np.where(p2d_3 < 1e-8)] = 1.0
        p2d[:, 2] = p2d_3
        p2d = np.around((p2d[:, :2] / p2d[:, 2:])).astype(np.int32)
        return p2d

    def ensure_dir(self, pth):
        if not os.path.exists(pth):
            os.system("mkdir -p %s" % pth)

    def draw_p2ds(self, img, p2ds, r=1, color=[(255, 0, 0)],alpha=0.4):
        # alpha = 0.4  # Transparency factor.
        # pick every 10 points
        # p2ds = p2ds[::10, :]
        overlay = img.copy()
        if type(color) == tuple:
            color = [color]
        if len(color) != p2ds.shape[0]:
            color = [color[0] for i in range(p2ds.shape[0])]
        h, w = img.shape[0], img.shape[1]
        for pt_2d, c in zip(p2ds, color):
            pt_2d[0] = np.clip(pt_2d[0], 0, w)
            pt_2d[1] = np.clip(pt_2d[1], 0, h)
            # img = cv2.circle(
            #     img, (pt_2d[0], pt_2d[1]), r, c, -1
            # )
            cv2.circle(
                overlay, (pt_2d[0], pt_2d[1]), r, c, -1
            )
            # alpha = 0.4  # Transparency factor.
            # # Following line overlays transparent rectangle over the image
            # img = cv2.addWeighted(img, alpha, img, 1 - alpha, 0)
        image_new = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        #return img
        return image_new

    def paste_p2ds(self, img, p2ds, color=[(255, 0, 0)]):
        if type(color) == tuple:
            color = [color]
        if len(color) != p2ds.shape[0]:
            color = [color[0] for i in range(p2ds.shape[0])]
        h, w = img.shape[0], img.shape[1]
        p2ds[:, 0] = np.clip(p2ds[:, 0], 0, w)
        p2ds[:, 1] = np.clip(p2ds[:, 1], 0, h)
        img[p2ds[:, 1], p2ds[:, 0]] = np.array(color)
        return img

    def draw_p2ds_lb(self, img, p2ds, label, r=1, color=(255, 0, 0)):
        h, w = img.shape[0], img.shape[1]
        for pt_2d, lb in zip(p2ds, label):
            pt_2d[0] = np.clip(pt_2d[0], 0, w)
            pt_2d[1] = np.clip(pt_2d[1], 0, h)
            color = self.get_label_color(lb)
            img = cv2.circle(
                img, (pt_2d[0], pt_2d[1]), r, color, -1
            )
        return img

    def quick_nrm_map(
        self, dpt, scale_to_mm, K=intrinsic_matrix['ycb_K1'], with_show=False
    ):
        dpt_mm = (dpt.copy() * scale_to_mm).astype(np.uint16)
        nrm_map = normalSpeed.depth_normal(dpt_mm, K[0][0], K[1][1], 5, 2000, 20, False)
        if with_show:
            nrm_map[np.isnan(nrm_map)] = 0.0
            nrm_map[np.isinf(nrm_map)] = 0.0
            show_nrm = ((nrm_map[:, :, :3] + 1.0) * 127).astype(np.uint8)
            return nrm_map, show_nrm
        return nrm_map

    def dpt_2_showdpt(self, dpt, scale2m=1.0):
        min_d, max_d = dpt[dpt > 0].min(), dpt.max()
        dpt[dpt > 0] = (dpt[dpt > 0]-min_d) / (max_d - min_d) * 255
        # dpt = (dpt / scale2m) / dpt.max() * 255 #127
        dpt = dpt.astype(np.uint8)
        im_color = cv2.applyColorMap(
            cv2.convertScaleAbs(dpt, alpha=1), cv2.COLORMAP_JET
        )
        return im_color

    def get_show_label_img(self, labels, mode=1):
        cls_ids = np.unique(labels)
        n_obj = np.max(cls_ids)
        if len(labels.shape) > 2:
            labels = labels[:, :, 0]
        h, w = labels.shape
        show_labels = np.zeros(
            (h, w, 3), dtype='uint8'
        )
        labels = labels.reshape(-1)
        show_labels = show_labels.reshape(-1, 3)
        for cls_id in cls_ids:
            if cls_id == 0:
                continue
            cls_color = np.array(
                self.get_label_color(cls_id, n_obj=n_obj, mode=mode)
            )
            show_labels[labels == cls_id, :] = cls_color
        show_labels = show_labels.reshape(h, w, 3)
        return show_labels
    def get_radius(self, cls_id, r=1):
        r = [
            1,
            1,
            2,
            2,
            2,
            2,
            1,
            2,
            2,
            1,
            2,
            1,
            2,
            2,
            2,
            2
        ]
        radius = r[cls_id]
        return radius
    def get_label_color(self, cls_id, n_obj=22, mode=0):
        if mode == 0:
            cls_color = [
                17, 255, 0,
                17, 255, 0,
                17, 255, 0,
                17, 255, 0,
                17, 255, 0,
                17, 255, 0,
                17, 255, 0,
                17, 255, 0,
                17, 255, 0,
                17, 255, 0,
                17, 255, 0,
                17, 255, 0,
                17, 255, 0,
                17, 255, 0,
                17, 255, 0, # lamp
                17, 255, 0,
                17, 255, 0,
                17, 255, 0,
                17, 255, 0,
                17, 255, 0,
            ]
            cls_color = np.array(cls_color).reshape(-1, 3)
            color = cls_color[cls_id]
            bgr = (int(color[0]), int(color[1]), int(color[2]))
        elif mode == 1:
            cls_color = [
                255, 255, 255,  # 0
                0, 127, 255,    # 180, 105, 255,   # 194, 194, 0,    # 1 # 194, 194, 0
                0, 255, 0,      # 2
                255, 0, 0,      # 3
                180, 105, 255, # 0, 255, 255,    # 4
                255, 0, 255,    # 5
                180, 105, 255,  # 128, 128, 0,    # 6
                128, 0, 0,      # 7
                0, 128, 0,      # 8
                185, 218, 255,# 0, 0, 255, # 0, 165, 255,    # 0, 0, 128,      # 9
                128, 128, 0,    # 10
                0, 0, 255,      # 11
                255, 0, 0,      # 12
                0, 194, 0,      # 13
                0, 194, 0,      # 14
                255, 255, 0,    # 15 # 0, 194, 194
                0, 0, 255, # 64, 64, 0,      # 16
                64, 0, 64,      # 17
                185, 218, 255,  # 0, 0, 64,       # 18
                0, 0, 255,      # 19
                0, 0, 255, # 0, 64, 0,       # 20
                0, 255, 255,# 0, 0, 192       # 21
            ]
            cls_color = np.array(cls_color).reshape(-1, 3)
            color = cls_color[cls_id]
            bgr = (int(color[0]), int(color[1]), int(color[2]))
        else:
            mul_col = 255 * 255 * 255 // n_obj * cls_id
            r, g, b= mul_col // 255 // 255, (mul_col // 255) % 255, mul_col % 255
            bgr = (int(r), int(g) , int(b))
        return bgr

    def dpt_2_cld(self, dpt, cam_scale, K):
        if len(dpt.shape) > 2:
            dpt = dpt[:, :, 0]
        msk_dp = dpt > 1e-6
        choose = msk_dp.flatten().nonzero()[0].astype(np.uint32)
        if len(choose) < 1:
            return None, None

        dpt_mskd = dpt.flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_mskd = self.xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_mskd = self.ymap.flatten()[choose][:, np.newaxis].astype(np.float32)

        pt2 = dpt_mskd / cam_scale
        cam_cx, cam_cy = K[0][2], K[1][2]
        cam_fx, cam_fy = K[0][0], K[1][1]
        pt0 = (ymap_mskd - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_mskd - cam_cy) * pt2 / cam_fy
        cld = np.concatenate((pt0, pt1, pt2), axis=1)
        return cld, choose

    def get_normal_map(self, nrm, choose):
        nrm_map = np.zeros((480, 640, 3), dtype=np.uint8)
        nrm = nrm[:, :3]
        nrm[np.isnan(nrm)] = 0.0
        nrm[np.isinf(nrm)] = 0.0
        nrm_color = ((nrm + 1.0) * 127).astype(np.uint8)
        nrm_map = nrm_map.reshape(-1, 3)
        nrm_map[choose, :] = nrm_color
        nrm_map = nrm_map.reshape((480, 640, 3))
        return nrm_map

    def get_rgb_pts_map(self, pts, choose):
        pts_map = np.zeros((480, 640, 3), dtype=np.uint8)
        pts = pts[:, :3]
        pts[np.isnan(pts)] = 0.0
        pts[np.isinf(pts)] = 0.0
        pts_color = pts.astype(np.uint8)
        pts_map = pts_map.reshape(-1, 3)
        pts_map[choose, :] = pts_color
        pts_map = pts_map.reshape((480, 640, 3))
        return pts_map

    def fill_missing(
            self, dpt, cam_scale, scale_2_80m, fill_type='multiscale',
            extrapolate=False, show_process=False, blur_type='bilateral'
    ):
        dpt = dpt / cam_scale * scale_2_80m
        projected_depth = dpt.copy()
        if fill_type == 'fast':
            final_dpt = depth_map_utils.fill_in_fast(
                projected_depth, extrapolate=extrapolate, blur_type=blur_type,
                # max_depth=2.0
            )
        elif fill_type == 'multiscale':
            final_dpt, process_dict = depth_map_utils.fill_in_multiscale(
                projected_depth, extrapolate=extrapolate, blur_type=blur_type,
                show_process=show_process,
                max_depth=3.0
            )
        else:
            raise ValueError('Invalid fill_type {}'.format(fill_type))
        dpt = final_dpt / scale_2_80m * cam_scale
        return dpt

    def rand_range(self, lo, hi):
        return random.random()*(hi-lo)+lo

    def get_ycb_ply_mdl(
        self, cls
    ):
        ply_pattern = os.path.join(
            self.config.ycb_root, '/models',
            '{}/textured.ply'
        )
        ply = PlyData.read(ply_pattern.format(cls, cls))
        data = ply.elements[0].data
        x = data['x']
        y = data['y']
        z = data['z']
        model = np.stack([x, y, z], axis=-1)
        return model

    def get_cls_name(self, cls, ds_type):
        if type(cls) is int:
            if ds_type == 'ycb':
                cls = self.ycb_cls_lst[cls - 1]
            else:
                cls = self.lm_cls_lst[cls - 1]
        return cls

    def ply_vtx(self, pth):
        f = open(pth)
        assert f.readline().strip() == "ply"
        f.readline()
        f.readline()
        N = int(f.readline().split()[-1])
        while f.readline().strip() != "end_header":
            continue
        pts = []
        for _ in range(N):
            pts.append(np.float32(f.readline().split()[:3]))
        return np.array(pts)

    def get_pointxyz(
        self, cls, ds_type='ycb'
    ):
        if ds_type == "ycb":
            cls = self.get_cls_name(cls, ds_type)
            if cls in self.ycb_cls_ptsxyz_dict.keys():
                return self.ycb_cls_ptsxyz_dict[cls]
            ptxyz_ptn = os.path.join(
                self.config.ycb_root, 'models',
                '{}/points.xyz'.format(cls),
            )
            pointxyz = np.loadtxt(ptxyz_ptn.format(cls), dtype=np.float32)
            self.ycb_cls_ptsxyz_dict[cls] = pointxyz
            return pointxyz
        else:
            ptxyz_pth = os.path.join(
                self.config.lm_root,'models',
                'obj_%02d.ply' % cls
            )
            pointxyz = self.ply_vtx(ptxyz_pth) / 1000.0
            dellist = [j for j in range(0, len(pointxyz))]
            dellist = random.sample(dellist, len(pointxyz) - 2000)
            pointxyz = np.delete(pointxyz, dellist, axis=0)
            self.lm_cls_ptsxyz_dict[cls] = pointxyz
            return pointxyz

    def get_pointxyz_cuda(
        self, cls, ds_type='ycb'
    ):
        if ds_type == "ycb":
            if cls in self.ycb_cls_ptsxyz_cuda_dict.keys():
                return self.ycb_cls_ptsxyz_cuda_dict[cls].clone()
            ptsxyz = self.get_pointxyz(cls, ds_type)
            ptsxyz_cu = torch.from_numpy(ptsxyz.astype(np.float32)).cuda()
            self.ycb_cls_ptsxyz_cuda_dict[cls] = ptsxyz_cu
            return ptsxyz_cu.clone()
        elif ds_type=='lab':
            if cls in self.lm_cls_ptsxyz_cuda_dict.keys():
                return self.lm_cls_ptsxyz_cuda_dict[cls].clone()
            ptsxyz = self.get_pointxyz(cls, ds_type)
            ptsxyz_cu = torch.from_numpy(ptsxyz.astype(np.float32)).cuda()
            self.lm_cls_ptsxyz_cuda_dict[cls] = ptsxyz_cu
            return ptsxyz_cu.clone()
        else:
            if cls in self.lm_cls_ptsxyz_cuda_dict.keys():
                return self.lm_cls_ptsxyz_cuda_dict[cls].clone()
            ptsxyz = self.get_pointxyz(cls, ds_type)
            ptsxyz_cu = torch.from_numpy(ptsxyz.astype(np.float32)).cuda()
            self.lm_cls_ptsxyz_cuda_dict[cls] = ptsxyz_cu
            return ptsxyz_cu.clone()

    def get_kps(
        self, cls, kp_type='farthest', ds_type='ycb', kp_pth=None
    ):
        if kp_pth:
            kps = np.loadtxt(kp_pth, dtype=np.float32)
            return kps
        if type(cls) is int:
            if ds_type == 'ycb':
                cls = self.ycb_cls_lst[cls - 1]
            elif ds_type=="linemod":
                cls = self.config.lm_id2obj_dict[cls]
            elif ds_type=="lab":
                cls = self.config.lm_id2obj_dict[cls]
            else:
                cls = self.config.lmo_id2obj_dict[cls]
        try:
            use_orbfps = self.config.use_orbfps
        except Exception:
            use_orbfps = False
        if ds_type == "ycb":
            if cls in self.ycb_cls_kps_dict.keys():
                return self.ycb_cls_kps_dict[cls].copy()
            if use_orbfps:
                kps_pth = self.config.kp_orbfps_ptn % (cls, self.config.n_keypoints)
            else:
                kps_pth = os.path.join(
                    self.config.ycb_fps_kps_dir, '{}/{}.txt'.format(cls, kp_type)
                )
            kps = np.loadtxt(kps_pth, dtype=np.float32)
            self.ycb_cls_kps_dict[cls] = kps
        elif ds_type=="linemod":
            if cls in self.lm_cls_kps_dict.keys():
                return self.lm_cls_kps_dict[cls].copy()
            if use_orbfps:
                kps_pth = self.config.kp_orbfps_ptn % (cls, self.config.n_keypoints)
            else:
                kps_pattern = os.path.join(
                    self.config.lm_kps_dir, "{}/{}.txt".format(cls, kp_type)
                )
                kps_pth = kps_pattern.format(cls)
            
            kps = np.loadtxt(kps_pth, dtype=np.float32)
            self.lm_cls_kps_dict[cls] = kps
        elif ds_type=="lab":
            if cls in self.lm_cls_kps_dict.keys():
                return self.lm_cls_kps_dict[cls].copy()
            if use_orbfps:
                kps_pth = self.config.kp_orbfps_ptn % (cls, self.config.n_keypoints)
            else:
                kps_pattern = os.path.join(
                    self.config.lm_kps_dir, "{}/{}.txt".format(cls, kp_type)
                )
                kps_pth = kps_pattern.format(cls)
            
            kps = np.loadtxt(kps_pth, dtype=np.float32)
            self.lm_cls_kps_dict[cls] = kps
        else:
            if cls in self.lmo_cls_kps_dict.keys():
                return self.lmo_cls_kps_dict[cls].copy()
            if use_orbfps:
                kps_pth = self.config.kp_orbfps_ptn % (cls, self.config.n_keypoints)
            else:
                kps_pattern = os.path.join(
                    self.config.lmo_kps_dir, "{}/{}.txt".format(cls, kp_type)
                )
                kps_pth = kps_pattern.format(cls)
            
            kps = np.loadtxt(kps_pth, dtype=np.float32)
            self.lmo_cls_kps_dict[cls] = kps
        return kps.copy()

    def get_ctr(self, cls, ds_type='ycb', ctr_pth=None):
        if ctr_pth:
            ctr = np.loadtxt(ctr_pth, dtype=np.float32)
            return ctr
        if type(cls) is int:
            if ds_type == 'ycb':
                cls = self.ycb_cls_lst[cls - 1]
            elif ds_type=="linemod":
                cls = self.config.lm_id2obj_dict[cls]
            elif ds_type=="lab":
                cls = self.config.lm_id2obj_dict[cls]
            else:
                cls = self.config.lmo_id2obj_dict[cls]
        if ds_type == "ycb":
            if cls in self.ycb_cls_ctr_dict.keys():
                return self.ycb_cls_ctr_dict[cls].copy()
            cor_pattern = os.path.join(
                self.config.kp_orbfps_dir, '{}_corners.txt'.format(cls),
            )
            cors = np.loadtxt(cor_pattern.format(cls), dtype=np.float32)
            ctr = cors.mean(0)
            self.ycb_cls_ctr_dict[cls] = ctr
        elif ds_type=="linemod":
            if cls in self.lm_cls_ctr_dict.keys():
                return self.lm_cls_ctr_dict[cls].copy()
            cor_pattern = os.path.join(
                self.config.kp_orbfps_dir, '{}_corners.txt'.format(cls),
            )
            cors = np.loadtxt(cor_pattern.format(cls), dtype=np.float32)
            ctr = cors.mean(0)
            self.lm_cls_ctr_dict[cls] = ctr
        elif ds_type=="lab":
            if cls in self.lm_cls_ctr_dict.keys():
                return self.lm_cls_ctr_dict[cls].copy()
            cor_pattern = os.path.join(
                self.config.kp_orbfps_dir, '{}_corners.txt'.format(cls),
            )
            cors = np.loadtxt(cor_pattern.format(cls), dtype=np.float32)
            ctr = cors.mean(0)
            self.lm_cls_ctr_dict[cls] = ctr
        else:
            if cls in self.lmo_cls_ctr_dict.keys():
                return self.lmo_cls_ctr_dict[cls].copy()
            cor_pattern = os.path.join(
                self.config.kp_orbfps_dir, '{}_corners.txt'.format(cls),
            )
            cors = np.loadtxt(cor_pattern.format(cls), dtype=np.float32)
            ctr = cors.mean(0)
            self.lmo_cls_ctr_dict[cls] = ctr
        return ctr.copy()

    def occlm_draw_points(self, img_id, folder_name, obj_id, obj_name, pred_RT, p3ds):
        
        pred_p3ds = torch.mm(p3ds, pred_RT[:, :3].transpose(1, 0)) + pred_RT[:, 3]
        # for cropping image
        # show_kp_img = cv2.imread('/workspace/DATA/Linemod_preprocessed/data/'+str(obj_id)+'/crop_rgb/'+img_id+'.png')
        
        #show_kp_img = cv2.imread(os.path.join('/workspace/REPO/pose_estimation/ffb6d/LineMod_Vis/ape',img_id+'.png'))
        show_kp_img = cv2.imread('/workspace/DATA/Occ_LineMod/test/000002/rgb/'+img_id.zfill(6)+'.png')
        pred_2ds = self.project_p3d(
            pred_p3ds.cpu().numpy(), 1000.0, K='linemod'
        )
        
        color = self.get_label_color(obj_id)
        # radius = self.get_radius(obj_id)
        radius = 1
        show_kp_img = self.draw_p2ds(show_kp_img, pred_2ds, r=radius, color=color,alpha=0.7)
        # imshow("kp: cls_id=%d" % cls_id, show_kp_img)
        if not os.path.exists(os.path.join('/workspace/REPO/pose_estimation/ffb6d/Occ_LineMod_Vis',obj_name)):
            os.makedirs(os.path.join('/workspace/REPO/pose_estimation/ffb6d/Occ_LineMod_Vis',obj_name))
        
        cv2.imwrite(os.path.join('/workspace/REPO/pose_estimation/ffb6d/Occ_LineMod_Vis',obj_name,img_id+'.png'), show_kp_img)
    
    def lm_draw_points_kp(self, img_id, folder_name, obj_id, obj_name, pred_RT, p3ds):
        
        pred_p3ds = p3ds
        # for cropping image
        # show_kp_img = cv2.imread('/workspace/DATA/Linemod_preprocessed/data/'+str(obj_id)+'/crop_rgb/'+img_id+'.png')
        
        #show_kp_img = cv2.imread(os.path.join('/workspace/REPO/pose_estimation/ffb6d/LineMod_Vis/ape',img_id+'.png'))
        show_kp_img = cv2.imread('/workspace/DATA/Linemod_preprocessed/data/'+str(obj_id).zfill(2)+'/rgb/'+img_id+'.png')
        pred_2ds = self.project_p3d(
            pred_p3ds.cpu().numpy(), 1000.0, K='linemod'
        )
        
        color = self.get_label_color(obj_id)
        # radius = self.get_radius(obj_id)
        radius = 3
        show_kp_img = self.draw_p2ds(show_kp_img, pred_2ds, r=radius, color=color,alpha=1)
        # imshow("kp: cls_id=%d" % cls_id, show_kp_img)
        if not os.path.exists(os.path.join('/workspace/REPO/pose_estimation/ffb6d/LineMod_Vis_kps',obj_name)):
            os.makedirs(os.path.join('/workspace/REPO/pose_estimation/ffb6d/LineMod_Vis_kps',obj_name))
        
        cv2.imwrite(os.path.join('/workspace/REPO/pose_estimation/ffb6d/LineMod_Vis_kps',obj_name,img_id+'.png'), show_kp_img)    
    
    def lm_draw_points(self, img_id, folder_name, obj_id, obj_name, pred_RT, p3ds):
         
        pred_p3ds = torch.mm(p3ds, pred_RT[:, :3].transpose(1, 0)) + pred_RT[:, 3]
        # for cropping image
        # show_kp_img = cv2.imread('/workspace/DATA/Linemod_preprocessed/data/'+str(obj_id)+'/crop_rgb/'+img_id+'.png')
        
        #show_kp_img = cv2.imread(os.path.join('/workspace/REPO/pose_estimation/ffb6d/LineMod_Vis/ape',img_id+'.png'))
        show_kp_img = cv2.imread('/workspace/DATA/Linemod_preprocessed/data/'+str(obj_id).zfill(2)+'/rgb/'+img_id+'.png')
        pred_2ds = self.project_p3d(
            pred_p3ds.cpu().numpy(), 1000.0, K='linemod'
        )
        
        color = self.get_label_color(obj_id)
        # radius = self.get_radius(obj_id)
        radius = 1
        show_kp_img = self.draw_p2ds(show_kp_img, pred_2ds, r=radius, color=color,alpha=0.7)
        # imshow("kp: cls_id=%d" % cls_id, show_kp_img)
        if not os.path.exists(os.path.join('/workspace/REPO/pose_estimation/ffb6d/LineMod_Vis',obj_name)):
            os.makedirs(os.path.join('/workspace/REPO/pose_estimation/ffb6d/LineMod_Vis',obj_name))
        
        cv2.imwrite(os.path.join('/workspace/REPO/pose_estimation/ffb6d/LineMod_Vis',obj_name,img_id+'.png'), show_kp_img)
    
    def lab_draw_points(self, output_path, input_path, cat_id, pred_RT, p3ds):
         
        pred_p3ds = torch.mm(p3ds, pred_RT[:, :3].transpose(1, 0)) + pred_RT[:, 3]
        
        show_kp_img = cv2.imread(input_path)
        pred_2ds = self.project_p3d(
            pred_p3ds.cpu().numpy(), 1000.0, K='lab'
        )
        
        color = self.get_label_color(cat_id)
        # radius = self.get_radius(obj_id)
        radius = 1
        show_kp_img = self.draw_p2ds(show_kp_img, pred_2ds, r=radius, color=color,alpha=0.7)
        # imshow("kp: cls_id=%d" % cls_id, show_kp_img)
        
        cv2.imwrite(output_path, show_kp_img)
        
        
    def depth2show(self, depth):
        show_depth = (depth / depth.max() * 256).astype("uint8")
        return show_depth
    def lm_draw_points_depth(self, img_id, folder_name, obj_id, obj_name, pred_RT, p3ds):
         
        pred_p3ds = torch.mm(p3ds, pred_RT[:, :3].transpose(1, 0)) + pred_RT[:, 3]
        # for cropping image
        # show_kp_img = cv2.imread('/workspace/DATA/Linemod_preprocessed/data/'+str(obj_id)+'/crop_rgb/'+img_id+'.png')
        

        show_kp_img = cv2.imread('/workspace/DATA/Linemod_preprocessed/data/'+str(obj_id).zfill(2)+'/depth/'+img_id+'.png')
        show_kp_img = self.depth2show(show_kp_img)
        pred_2ds = self.project_p3d(pred_p3ds.cpu().numpy(), 1000.0, K='linemod')
        color = self.get_label_color(obj_id)        
        show_kp_img = self.draw_p2ds(show_kp_img, pred_2ds, r=1, color=color)
        # imshow("kp: cls_id=%d" % cls_id, show_kp_img)
        if not os.path.exists(os.path.join('/workspace/REPO/pose_estimation/ffb6d/LineMod_Vis',obj_name)):
            os.makedirs(os.path.join('/workspace/REPO/pose_estimation/ffb6d/LineMod_Vis',obj_name))
        
        cv2.imwrite(os.path.join('/workspace/REPO/pose_estimation/ffb6d/LineMod_Vis',obj_name,img_id+'_depth.png'), show_kp_img)
    
    def save_points(self, img_id, folder_name, obj_name, pred_RT, gt_RT, p3ds):
        img_id = img_id.cpu().detach().numpy()
        img_id = str(int(img_id)).zfill(4)
        pred_p3ds = torch.mm(p3ds, pred_RT[:, :3].transpose(1, 0)) + pred_RT[:, 3]
        gt_p3ds = torch.mm(p3ds, gt_RT[:, :3].transpose(1, 0)) + gt_RT[:, 3]
        
        pred_p3ds = pred_p3ds.cpu().detach().numpy()
        gt_p3ds = gt_p3ds.cpu().detach().numpy()
    
        np.savetxt('/workspace/REPO/pose_estimation/ffb6d/train_log/'+folder_name+'/'+obj_name+'/eval_results_gt/'+img_id+'_pred.txt', pred_p3ds)
        np.savetxt('/workspace/REPO/pose_estimation/ffb6d/train_log/'+folder_name+'/'+obj_name+'/eval_results_gt/'+img_id+'_gt.txt', gt_p3ds)
    
    def cal_auc(self, add_dis, max_dis=0.1):
        D = np.array(add_dis)
        D[np.where(D > max_dis)] = np.inf
        D = np.sort(D)
        n = len(add_dis)
        acc = np.cumsum(np.ones((1,n)), dtype=np.float32) / n
        aps = VOCap(D, acc)
        return aps * 100
    
    def cal_add_cuda_icp(
        self, pred_RT, gt_RT, p3ds
    ):
        pred_p3ds = torch.mm(p3ds, pred_RT[:, :3].transpose(1, 0)) + pred_RT[:, 3]
        gt_p3ds = torch.mm(p3ds, gt_RT[:, :3].transpose(1, 0)) + gt_RT[:, 3]
        # apply ICP 
        icp_R, icp_t = icp.icp(pred_p3ds, gt_p3ds, tolerance=0.000001)
        icp_p3ds = torch.mm(pred_p3ds, icp_R[:, :3].transpose(1, 0)) + icp_t[:, 3]
        dis = torch.norm(icp_p3ds - gt_p3ds, dim=1)
        return torch.mean(dis)

    def cal_adds_cuda_icp(
        self, pred_RT, gt_RT, p3ds
    ):
        N, _ = p3ds.size()
        pd = torch.mm(p3ds, pred_RT[:, :3].transpose(1, 0)) + pred_RT[:, 3]
        pd_s = pd.view(1, N, 3).repeat(N, 1, 1)
        gt = torch.mm(p3ds, gt_RT[:, :3].transpose(1, 0)) + gt_RT[:, 3]
        gt_s = gt.view(N, 1, 3).repeat(1, N, 1)
        # apply ICP 
        icp_R, icp_t = icp.icp(pd, gt, tolerance=0.000001)
        icp_p3ds = torch.mm(pd, icp_R[:, :3].transpose(1, 0)) + icp_t[:, 3]
        icp_p3ds = icp_p3ds.view(1, N, 3).repeat(N, 1, 1)
        dis = torch.norm(icp_p3ds - gt_s, dim=2)
        mdis = torch.min(dis, dim=1)[0]
        return torch.mean(mdis)
    def cal_add_cuda(
        self, pred_RT, gt_RT, p3ds
    ):
        pred_p3ds = torch.mm(p3ds, pred_RT[:, :3].transpose(1, 0)) + pred_RT[:, 3]
        gt_p3ds = torch.mm(p3ds, gt_RT[:, :3].transpose(1, 0)) + gt_RT[:, 3]
        
        dis = torch.norm(pred_p3ds - gt_p3ds, dim=1)
        return torch.mean(dis)

    def cal_adds_cuda(
        self, pred_RT, gt_RT, p3ds
    ):
        N, _ = p3ds.size()
        pd = torch.mm(p3ds, pred_RT[:, :3].transpose(1, 0)) + pred_RT[:, 3]
        pd = pd.view(1, N, 3).repeat(N, 1, 1)
        gt = torch.mm(p3ds, gt_RT[:, :3].transpose(1, 0)) + gt_RT[:, 3]
        gt = gt.view(N, 1, 3).repeat(1, N, 1)
        dis = torch.norm(pd - gt, dim=2)
        mdis = torch.min(dis, dim=1)[0]
        return torch.mean(mdis)

    def best_fit_transform_torch(self, A, B):
        '''
        Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
        Input:
            A: Nxm numpy array of corresponding points, usually points on mdl
            B: Nxm numpy array of corresponding points, usually points on camera axis
        Returns:
        T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
        R: mxm rotation matrix
        t: mx1 translation vector
        '''
        assert A.size() == B.size()
        # get number of dimensions
        m = A.size()[1]
        # translate points to their centroids
        centroid_A = torch.mean(A, dim=0)
        centroid_B = torch.mean(B, dim=0)
        AA = A - centroid_A
        BB = B - centroid_B
        # rotation matirx
        H = torch.mm(AA.transpose(1, 0), BB)
        U, S, Vt = torch.svd(H)
        R = torch.mm(Vt.transpose(1, 0), U.transpose(1, 0))
        # special reflection case
        if torch.det(R) < 0:
            Vt[m-1, :] *= -1
            R = torch.mm(Vt.transpose(1, 0), U.transpose(1, 0))
        # translation
        t = centroid_B - torch.mm(R, centroid_A.view(3, 1))[:, 0]
        T = torch.zeros(3, 4).cuda()
        T[:, :3] = R
        T[:, 3] = t
        return  T

    def best_fit_transform(self, A, B):
        return best_fit_transform(A, B)


if __name__ == "__main__":

    pass
# vim: ts=4 sw=4 sts=4 expandtab
