#!/usr/bin/env python3
import os
import time
import torch
import numpy as np
import cv2
import pickle as pkl
import concurrent.futures
from config.options import BaseOptions
from config.common import Config
from utils.basic_utils import Basic_Utils
from utils.meanshift_pytorch import MeanShiftTorch
from utils.icp import icp
try:
    from neupeak.utils.webcv2 import imshow, waitKey
except Exception:
    from cv2 import imshow, waitKey
import pdb
opt = BaseOptions().parse()
if opt.dataset_name=='linemod':
    config_lm = Config(ds_name="linemod", cls_type=opt.linemod_cls)
    bs_utils_lm = Basic_Utils(config_lm)
elif opt.dataset_name=='lab':
    config_lab = Config(ds_name="lab", cls_type=opt.linemod_cls)
    bs_utils_lab = Basic_Utils(config_lab)
elif opt.dataset_name=='ycb':
    config = Config(ds_name='ycb')
    bs_utils = Basic_Utils(config)
    cls_lst = config.ycb_cls_lst
else:
    config_lmo = Config(ds_name="occlusion_linemod", cls_type=opt.occ_linemod_cls)
    bs_utils_lmo = Basic_Utils(config_lmo)



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


# ###############################YCB Evaluation###############################
def cal_frame_poses(
    pcld, mask, ctr_of, pred_kp_of, use_ctr, n_cls, use_ctr_clus_flter,
    gt_kps, gt_ctrs, debug=False, kp_type='farthest'
):
    """
    Calculates pose parameters by 3D keypoints & center points voting to build
    the 3D-3D corresponding then use least-squares fitting to get the pose parameters.
    """
    n_kps, n_pts, _ = pred_kp_of.size()
    pred_ctr = pcld - ctr_of[0]
    pred_kp = pcld.view(1, n_pts, 3).repeat(n_kps, 1, 1) - pred_kp_of

    radius = 0.04
    if use_ctr:
        cls_kps = torch.zeros(n_cls, n_kps+1, 3).cuda()
    else:
        cls_kps = torch.zeros(n_cls, n_kps, 3).cuda()

    # Use center clustering filter to improve the predicted mask.
    pred_cls_ids = np.unique(mask[mask > 0].contiguous().cpu().numpy())
    if use_ctr_clus_flter:
        ctrs = []
        for icls, cls_id in enumerate(pred_cls_ids):
            cls_msk = (mask == cls_id)
            ms = MeanShiftTorch(bandwidth=radius)
            ctr, ctr_labels = ms.fit(pred_ctr[cls_msk, :])
            ctrs.append(ctr.detach().contiguous().cpu().numpy())
        try:
            ctrs = torch.from_numpy(np.array(ctrs).astype(np.float32)).cuda()
            n_ctrs, _ = ctrs.size()
            pred_ctr_rp = pred_ctr.view(n_pts, 1, 3).repeat(1, n_ctrs, 1)
            ctrs_rp = ctrs.view(1, n_ctrs, 3).repeat(n_pts, 1, 1)
            ctr_dis = torch.norm((pred_ctr_rp - ctrs_rp), dim=2)
            min_dis, min_idx = torch.min(ctr_dis, dim=1)
            msk_closest_ctr = torch.LongTensor(pred_cls_ids).cuda()[min_idx]
            new_msk = mask.clone()
            for cls_id in pred_cls_ids:
                if cls_id == 0:
                    break
                min_msk = min_dis < config.ycb_r_lst[cls_id-1] * 0.8
                update_msk = (mask > 0) & (msk_closest_ctr == cls_id) & min_msk
                new_msk[update_msk] = msk_closest_ctr[update_msk]
            mask = new_msk
        except Exception:
            pass

    # 3D keypoints voting and least squares fitting for pose parameters estimation.
    pred_pose_lst = []
    pred_kps_lst = []
    for icls, cls_id in enumerate(pred_cls_ids):
        if cls_id == 0:
            break
        cls_msk = mask == cls_id
        if cls_msk.sum() < 1:
            pred_pose_lst.append(np.identity(4)[:3, :])
            pred_kps_lst.append(np.zeros((n_kps+1, 3)))
            continue

        cls_voted_kps = pred_kp[:, cls_msk, :]
        ms = MeanShiftTorch(bandwidth=radius)
        ctr, ctr_labels = ms.fit(pred_ctr[cls_msk, :])
        if ctr_labels.sum() < 1:
            ctr_labels[0] = 1
        if use_ctr:
            cls_kps[cls_id, n_kps, :] = ctr

        if use_ctr_clus_flter:
            in_pred_kp = cls_voted_kps[:, ctr_labels, :]
        else:
            in_pred_kp = cls_voted_kps

        for ikp, kps3d in enumerate(in_pred_kp):
            cls_kps[cls_id, ikp, :], _ = ms.fit(kps3d)

        # visualize
        if debug:
            show_kp_img = np.zeros((480, 640, 3), np.uint8)
            kp_2ds = bs_utils.project_p3d(cls_kps[cls_id].cpu().numpy(), 1000.0)
            color = bs_utils.get_label_color(cls_id.item())
            show_kp_img = bs_utils.draw_p2ds(show_kp_img, kp_2ds, r=3, color=color)
            imshow("kp: cls_id=%d" % cls_id, show_kp_img)
            waitKey(0)

        # Get mesh keypoint & center point in the object coordinate system.
        # If you use your own objects, check that you load them correctly.
        mesh_kps = bs_utils.get_kps(cls_lst[cls_id-1], kp_type=kp_type)
        if use_ctr:
            mesh_ctr = bs_utils.get_ctr(cls_lst[cls_id-1]).reshape(1, 3)
            mesh_kps = np.concatenate((mesh_kps, mesh_ctr), axis=0)
        pred_kpc = cls_kps[cls_id].squeeze().contiguous().cpu().numpy()
        pred_RT = best_fit_transform(mesh_kps, pred_kpc)
        pred_kps_lst.append(pred_kpc)
        pred_pose_lst.append(pred_RT)

    return (pred_cls_ids, pred_pose_lst, pred_kps_lst)


def eval_metric(
    cls_ids, pred_pose_lst, pred_cls_ids, RTs, mask, label,
    gt_kps, gt_ctrs, pred_kpc_lst
):
    n_cls = config.n_classes
    cls_add_dis = [list() for i in range(n_cls)]
    cls_adds_dis = [list() for i in range(n_cls)]
    cls_kp_err = [list() for i in range(n_cls)]
    for icls, cls_id in enumerate(cls_ids):
        if cls_id == 0:
            break

        gt_kp = gt_kps[icls].contiguous().cpu().numpy()

        cls_idx = np.where(pred_cls_ids == cls_id[0].item())[0]
        if len(cls_idx) == 0:
            pred_RT = torch.zeros(3, 4).cuda()
            pred_kp = np.zeros(gt_kp.shape)
        else:
            pred_RT = pred_pose_lst[cls_idx[0]]
            pred_kp = pred_kpc_lst[cls_idx[0]][:-1, :]
            pred_RT = torch.from_numpy(pred_RT.astype(np.float32)).cuda()
        kp_err = np.linalg.norm(gt_kp-pred_kp, axis=1).mean()
        cls_kp_err[cls_id].append(kp_err)
        gt_RT = RTs[icls]
        mesh_pts = bs_utils.get_pointxyz_cuda(cls_lst[cls_id-1]).clone()
        add = bs_utils.cal_add_cuda(pred_RT, gt_RT, mesh_pts)
        adds = bs_utils.cal_adds_cuda(pred_RT, gt_RT, mesh_pts)
        cls_add_dis[cls_id].append(add.item())
        cls_adds_dis[cls_id].append(adds.item())
        cls_add_dis[0].append(add.item())
        cls_adds_dis[0].append(adds.item())

    return (cls_add_dis, cls_adds_dis, cls_kp_err)


def eval_one_frame_pose(
    item
):
    pcld, mask, ctr_of, pred_kp_of, RTs, cls_ids, use_ctr, n_cls, \
        min_cnt, use_ctr_clus_flter, label, epoch, ibs, gt_kps, gt_ctrs, kp_type = item

    pred_cls_ids, pred_pose_lst, pred_kpc_lst = cal_frame_poses(
        pcld, mask, ctr_of, pred_kp_of, use_ctr, n_cls, use_ctr_clus_flter,
        gt_kps, gt_ctrs, kp_type=kp_type
    )

    cls_add_dis, cls_adds_dis, cls_kp_err = eval_metric(
        cls_ids, pred_pose_lst, pred_cls_ids, RTs, mask, label, gt_kps, gt_ctrs,
        pred_kpc_lst
    )
    return (cls_add_dis, cls_adds_dis, pred_cls_ids, pred_pose_lst, cls_kp_err)

# ###############################End YCB Evaluation###############################

# ################################Occlusion LineMod Evaluation########################
def get_linemod_to_occlusion_transformation(object_name):
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
def cal_frame_poses_occlm(
    pcld, mask, ctr_of, pred_kp_of, gt_kps, gt_ctr, use_ctr, n_cls, use_ctr_clus_flter, obj_id,
    debug=False
):
    """
    Calculates pose parameters by 3D keypoints & center points voting to build
    the 3D-3D corresponding then use least-squares fitting to get the pose parameters.
    """
    
    n_kps, n_pts, _ = pred_kp_of.size()
    pred_ctr = pcld - ctr_of[0]
    pred_kp = pcld.view(1, n_pts, 3).repeat(n_kps, 1, 1) - pred_kp_of

    radius = 0.04
    if use_ctr:
        cls_kps = torch.zeros(n_cls, n_kps+1, 3).cuda()
    else:
        cls_kps = torch.zeros(n_cls, n_kps, 3).cuda()

    pred_pose_lst = []
    cls_id = 1
    cls_msk = mask == cls_id
    if cls_msk.sum() < 1:
        pred_pose_lst.append(np.identity(4)[:3, :])
    else:
        cls_voted_kps = pred_kp[:, cls_msk, :]
        # for i in range(8):   
        #     np.savetxt(str(i)+'_pred_kps_rgb.txt', cls_voted_kps[i,:,:].squeeze().cpu())
        ms = MeanShiftTorch(bandwidth=radius)
        ctr, ctr_labels = ms.fit(pred_ctr[cls_msk, :])
        if ctr_labels.sum() < 1:
            ctr_labels[0] = 1
        if use_ctr:
            cls_kps[cls_id, n_kps, :] = ctr

        if use_ctr_clus_flter:
            in_pred_kp = cls_voted_kps[:, ctr_labels, :]
        else:
            in_pred_kp = cls_voted_kps

        for ikp, kps3d in enumerate(in_pred_kp):
            cls_kps[cls_id, ikp, :], _ = ms.fit(kps3d)
        
        # visualize
        # if True:
        #     import pdb; pdb.set_trace()
        #     show_kp_img = np.zeros((480, 640, 3), np.uint8)
        #     kp_2ds = bs_utils_lm.project_p3d(
        #         cls_kps[cls_id].cpu().numpy(), 1000.0, K='linemod'
        #     )
        #     color = (0, 0, 255)  # bs_utils.get_label_color(cls_id.item())
        #     show_kp_img = bs_utils_lm.draw_p2ds(show_kp_img, kp_2ds, r=3, color=color)
        #     # imshow("kp: cls_id=%d" % cls_id, show_kp_img)
        #     cv2.imwrite('/workspace/REPO/pose_estimation/ffb6d/train_log/lm_1_pseudo/phone/eval_results/test.png', show_kp_img)
        #     waitKey(0)
        
        mesh_kps = bs_utils_lmo.get_kps(obj_id, ds_type="occlusion_linemod")
        if use_ctr:
            mesh_ctr = bs_utils_lmo.get_ctr(obj_id, ds_type="occlusion_linemod").reshape(1, 3)
            mesh_kps = np.concatenate((mesh_kps, mesh_ctr), axis=0)
        # mesh_kps = torch.from_numpy(mesh_kps.astype(np.float32)).cuda()
        pred_RT = best_fit_transform(
            mesh_kps,
            cls_kps[cls_id].squeeze().contiguous().cpu().numpy()
        )
        pred_pose_lst.append(pred_RT)
        
    return pred_pose_lst, cls_kps[cls_id].squeeze().contiguous().cpu().numpy(), gt_kps[0].squeeze().contiguous().cpu().numpy(), gt_ctr[0].squeeze().contiguous().cpu().numpy()

def eval_metric_occlm(cls_ids, pred_pose_lst, RTs, mask, label, obj_id):
    n_cls = config_lmo.n_classes
    cls_add_dis = [list() for i in range(n_cls)]
    cls_adds_dis = [list() for i in range(n_cls)]

    pred_RT = pred_pose_lst[0]
    pred_RT = torch.from_numpy(pred_RT.astype(np.float32)).cuda()
    gt_RT = RTs[0]
    mesh_pts = bs_utils_lmo.get_pointxyz_cuda(obj_id, ds_type="occlusion_linemod").clone()
    
    # Check points transformed by predicted pose and GT pose, projecting them to 2D images
    # bs_utils_lm.draw_points(img_id, opt.wandb_name, obj_id, opt.linemod_cls, pred_RT, mesh_pts)
    
    # Save points transformed by predicted pose and GT pose  
    # bs_utils_lm.save_points(img_id, opt.wandb_name, opt.linemod_cls, pred_RT, gt_RT, mesh_pts)
    
    add = bs_utils_lmo.cal_add_cuda(pred_RT, gt_RT, mesh_pts)
    adds = bs_utils_lmo.cal_adds_cuda(pred_RT, gt_RT, mesh_pts)
    # print("obj_id:", obj_id, add, adds)
    # cls_add_dis[obj_id].append(add.item())
    # cls_adds_dis[obj_id].append(adds.item())
    
    cls_add_dis[0].append(add.item())
    cls_adds_dis[0].append(adds.item())

    return (cls_add_dis, cls_adds_dis)

def eval_metric_occlm_vis(img_id, cls_ids, pred_pose_lst, RTs, mask, label, obj_id):
    n_cls = config_lmo.n_classes
    cls_add_dis = [list() for i in range(n_cls)]
    cls_adds_dis = [list() for i in range(n_cls)]

    pred_RT = pred_pose_lst[0]
    pred_RT = torch.from_numpy(pred_RT.astype(np.float32)).cuda()
    gt_RT = RTs[0]
    mesh_pts = bs_utils_lmo.get_pointxyz_cuda(obj_id, ds_type="occlusion_linemod").clone()
    mesh_pts = mesh_pts[::2]
    # Check points transformed by predicted pose and GT pose, projecting them to 2D images
    bs_utils_lmo.occlm_draw_points(img_id, opt.wandb_name, obj_id, opt.occ_linemod_cls, pred_RT, mesh_pts)

    # Save points transformed by predicted pose and GT pose  
    # bs_utils_lm.save_points(img_id, opt.wandb_name, opt.linemod_cls, pred_RT, gt_RT, mesh_pts)

    # print("obj_id:", obj_id, add, adds)
    # cls_add_dis[obj_id].append(add.item())
    # cls_adds_dis[obj_id].append(adds.item())

    return (pred_RT, gt_RT)

# ###############################LineMOD Evaluation###############################

def cal_frame_poses_lm(
    pcld, mask, ctr_of, pred_kp_of, gt_kps, gt_ctr, use_ctr, n_cls, use_ctr_clus_flter, obj_id,
    debug=False
):
    """
    Calculates pose parameters by 3D keypoints & center points voting to build
    the 3D-3D corresponding then use least-squares fitting to get the pose parameters.
    """
    
    n_kps, n_pts, _ = pred_kp_of.size()
    pred_ctr = pcld - ctr_of[0]
    pred_kp = pcld.view(1, n_pts, 3).repeat(n_kps, 1, 1) - pred_kp_of

    radius = 0.04
    if use_ctr:
        cls_kps = torch.zeros(n_cls, n_kps+1, 3).cuda()
    else:
        cls_kps = torch.zeros(n_cls, n_kps, 3).cuda()

    pred_pose_lst = []
    cls_id = 1
    cls_msk = mask == cls_id
    if cls_msk.sum() < 1:
        pred_pose_lst.append(np.identity(4)[:3, :])
    else:
        cls_voted_kps = pred_kp[:, cls_msk, :]
        # for i in range(8):   
        #     np.savetxt(str(i)+'_pred_kps_rgb.txt', cls_voted_kps[i,:,:].squeeze().cpu())
        ms = MeanShiftTorch(bandwidth=radius)
        ctr, ctr_labels = ms.fit(pred_ctr[cls_msk, :])
        if ctr_labels.sum() < 1:
            ctr_labels[0] = 1
        if use_ctr:
            cls_kps[cls_id, n_kps, :] = ctr

        if use_ctr_clus_flter:
            in_pred_kp = cls_voted_kps[:, ctr_labels, :]
        else:
            in_pred_kp = cls_voted_kps

        for ikp, kps3d in enumerate(in_pred_kp):
            cls_kps[cls_id, ikp, :], _ = ms.fit(kps3d)
        
        # visualize
        # if True:
        #     import pdb; pdb.set_trace()
        #     show_kp_img = np.zeros((480, 640, 3), np.uint8)
        #     kp_2ds = bs_utils_lm.project_p3d(
        #         cls_kps[cls_id].cpu().numpy(), 1000.0, K='linemod'
        #     )
        #     color = (0, 0, 255)  # bs_utils.get_label_color(cls_id.item())
        #     show_kp_img = bs_utils_lm.draw_p2ds(show_kp_img, kp_2ds, r=3, color=color)
        #     # imshow("kp: cls_id=%d" % cls_id, show_kp_img)
        #     cv2.imwrite('/workspace/REPO/pose_estimation/ffb6d/train_log/lm_1_pseudo/phone/eval_results/test.png', show_kp_img)
        #     waitKey(0)
        
        mesh_kps = bs_utils_lm.get_kps(obj_id, ds_type="linemod")
        if use_ctr:
            mesh_ctr = bs_utils_lm.get_ctr(obj_id, ds_type="linemod").reshape(1, 3)
            mesh_kps = np.concatenate((mesh_kps, mesh_ctr), axis=0)
        # mesh_kps = torch.from_numpy(mesh_kps.astype(np.float32)).cuda()
        pred_RT = best_fit_transform(
            mesh_kps,
            cls_kps[cls_id].squeeze().contiguous().cpu().numpy()
        )
        pred_pose_lst.append(pred_RT)
        
    return pred_pose_lst, cls_kps[cls_id].squeeze().contiguous().cpu().numpy(), gt_kps[0].squeeze().contiguous().cpu().numpy(), gt_ctr[0].squeeze().contiguous().cpu().numpy()

def cal_frame_poses_lab(
    cls_id, pcld, mask, pred_ctr_of, pred_kp_of, use_ctr, n_cls, use_ctr_clus_flter, obj_id,
    debug=False
):
    """
    Calculates pose parameters by 3D keypoints & center points voting to build
    the 3D-3D corresponding then use least-squares fitting to get the pose parameters.
    """
     
    n_kps, n_pts, _ = pred_kp_of.size()
    pred_ctr = pcld - pred_ctr_of[0]
    pred_kp = pcld.view(1, n_pts, 3).repeat(n_kps, 1, 1) - pred_kp_of

    radius = 0.04
    if use_ctr:
        cls_kps = torch.zeros(n_cls, n_kps+1, 3).cuda()
    else:
        cls_kps = torch.zeros(n_cls, n_kps, 3).cuda()

    pred_pose_lst = []
    cls_id_single = 1
    cls_msk = mask == cls_id_single
    if cls_msk.sum() < 1:
        pred_pose_lst.append(np.identity(4)[:3, :])
    else:
        cls_voted_kps = pred_kp[:, cls_msk, :]
        # for i in range(8):   
        #     np.savetxt(str(i)+'_pred_kps_rgb.txt', cls_voted_kps[i,:,:].squeeze().cpu())
        ms = MeanShiftTorch(bandwidth=radius)
        ctr, ctr_labels = ms.fit(pred_ctr[cls_msk, :])
        if ctr_labels.sum() < 1:
            ctr_labels[0] = 1
        if use_ctr:
            cls_kps[cls_id_single, n_kps, :] = ctr

        if use_ctr_clus_flter:
            in_pred_kp = cls_voted_kps[:, ctr_labels, :]
        else:
            in_pred_kp = cls_voted_kps

        for ikp, kps3d in enumerate(in_pred_kp):
            cls_kps[cls_id_single, ikp, :], _ = ms.fit(kps3d)
        
        # visualize predicted keypoints
        # if True:
        #     import pdb; pdb.set_trace()
        #     show_kp_img = np.zeros((480, 640, 3), np.uint8)
        #     kp_2ds = bs_utils_lm.project_p3d(
        #         cls_kps[cls_id].cpu().numpy(), 1000.0, K='linemod'
        #     )
        #     color = (0, 0, 255)  # bs_utils.get_label_color(cls_id.item())
        #     show_kp_img = bs_utils_lm.draw_p2ds(show_kp_img, kp_2ds, r=3, color=color)
        #     # imshow("kp: cls_id=%d" % cls_id, show_kp_img)
        #     cv2.imwrite('/workspace/REPO/pose_estimation/ffb6d/train_log/lm_1_pseudo/phone/eval_results/test.png', show_kp_img)
        #     waitKey(0)
        
        mesh_kps = bs_utils_lab.get_kps(cls_id, ds_type="lab")
        if use_ctr:
            mesh_ctr = bs_utils_lab.get_ctr(cls_id, ds_type="lab").reshape(1, 3)
            mesh_kps = np.concatenate((mesh_kps, mesh_ctr), axis=0)
        # mesh_kps = torch.from_numpy(mesh_kps.astype(np.float32)).cuda()
        pred_RT = best_fit_transform(
            mesh_kps,
            cls_kps[cls_id_single].squeeze().contiguous().cpu().numpy()
        )
        
        # visualize projected object
        if False:
            
            mesh_pts = bs_utils_lab.get_pointxyz_cuda(cls_id, ds_type="lab").clone()
            mesh_pts = mesh_pts[::2]
            show_kp_img = np.zeros((480, 640, 3), np.uint8)
            output_path = '/workspace/DATA/LabROS/all_vis_'+str(cls_id)+'.png'
            input_path = '/workspace/DATA/LabROS/data/all.jpg'
            
            kp_2ds = bs_utils_lab.lab_draw_points(
                output_path, input_path, cls_id, torch.from_numpy(pred_RT).double(), mesh_pts.cpu().double()
            )
            
            waitKey(0)
        
    return pred_RT

def eval_metric_lm_vis(img_id, cls_ids, pred_pose_lst, RTs, mask, label, obj_id, pred_kp, gt_kp, gt_ctr):
    
    n_cls = config_lm.n_classes
    # cls_add_dis = [list() for i in range(n_cls)]
    # cls_adds_dis = [list() for i in range(n_cls)]

    pred_RT = pred_pose_lst[0]
    pred_RT = torch.from_numpy(pred_RT.astype(np.float32)).cuda()
    gt_RT = RTs[0]
    mesh_pts = bs_utils_lm.get_pointxyz_cuda(obj_id, ds_type="linemod").clone()
    mesh_pts = mesh_pts[::2]
    
    pred_kp = torch.from_numpy(pred_kp.astype(np.float32)).cuda()
    gt_kp = torch.from_numpy(gt_kp.astype(np.float32)).cuda()
    gt_ctr = torch.from_numpy(gt_ctr.astype(np.float32)).cuda()
    gt_ctr = torch.unsqueeze(gt_ctr,0)
    gt_kps = torch.cat((gt_kp,gt_ctr),dim=0)
    # Check points transformed by predicted pose and GT pose, projecting them to 2D images
    bs_utils_lm.lm_draw_points(img_id, opt.wandb_name, obj_id, opt.linemod_cls, pred_RT, mesh_pts)
    # bs_utils_lm.lm_draw_points_kp(img_id, opt.wandb_name, obj_id, opt.linemod_cls, pred_RT, gt_kps)
    # Save points transformed by predicted pose and GT pose  
    # bs_utils_lm.save_points(img_id, opt.wandb_name, opt.linemod_cls, pred_RT, gt_RT, mesh_pts)
    
    # add = bs_utils_lm.cal_add_cuda(pred_RT, gt_RT, mesh_pts)
    # adds = bs_utils_lm.cal_adds_cuda(pred_RT, gt_RT, mesh_pts)
    # print("obj_id:", obj_id, add, adds)
    # cls_add_dis[obj_id].append(add.item())
    # cls_adds_dis[obj_id].append(adds.item())
    
    return (pred_RT, gt_RT)

def eval_metric_lm(img_id, cls_ids, pred_pose_lst, RTs, mask, label, obj_id):
    
    n_cls = config_lm.n_classes
    cls_add_dis = [list() for i in range(n_cls)]
    cls_adds_dis = [list() for i in range(n_cls)]

    pred_RT = pred_pose_lst[0]
    pred_RT = torch.from_numpy(pred_RT.astype(np.float32)).cuda()
    gt_RT = RTs[0]
    mesh_pts = bs_utils_lm.get_pointxyz_cuda(obj_id, ds_type="linemod").clone()
    
    # Check points transformed by predicted pose and GT pose, projecting them to 2D images
    # bs_utils_lm.lm_draw_points(img_id, opt.wandb_name, obj_id, opt.linemod_cls, pred_RT, mesh_pts)
    
    # Save points transformed by predicted pose and GT pose  
    # bs_utils_lm.save_points(img_id, opt.wandb_name, opt.linemod_cls, pred_RT, gt_RT, mesh_pts)
    
    add = bs_utils_lm.cal_add_cuda(pred_RT, gt_RT, mesh_pts)
    adds = bs_utils_lm.cal_adds_cuda(pred_RT, gt_RT, mesh_pts)
    # print("obj_id:", obj_id, add, adds)
    # cls_add_dis[obj_id].append(add.item())
    # cls_adds_dis[obj_id].append(adds.item())
    
    cls_add_dis[0].append(add.item())
    cls_adds_dis[0].append(adds.item())

    return (cls_add_dis, cls_adds_dis)
def eval_metric_lm_icp(img_id, cls_ids, pred_pose_lst, RTs, mask, label, obj_id):
    
    n_cls = config_lm.n_classes
    cls_add_dis = [list() for i in range(n_cls)]
    cls_adds_dis = [list() for i in range(n_cls)]

    pred_RT = pred_pose_lst[0]
    pred_RT = torch.from_numpy(pred_RT.astype(np.float32)).cuda()
    gt_RT = RTs[0]
    mesh_pts = bs_utils_lm.get_pointxyz_cuda(obj_id, ds_type="linemod").clone()
    
    # Check points transformed by predicted pose and GT pose, projecting them to 2D images
    # bs_utils_lm.lm_draw_points(img_id, opt.wandb_name, obj_id, opt.linemod_cls, pred_RT, mesh_pts)
    
    # Save points transformed by predicted pose and GT pose  
    # bs_utils_lm.save_points(img_id, opt.wandb_name, opt.linemod_cls, pred_RT, gt_RT, mesh_pts)
    
    add = bs_utils_lm.cal_add_cuda_icp(pred_RT, gt_RT, mesh_pts)
    adds = bs_utils_lm.cal_adds_cuda_icp(pred_RT, gt_RT, mesh_pts)
    # print("obj_id:", obj_id, add, adds)
    # cls_add_dis[obj_id].append(add.item())
    # cls_adds_dis[obj_id].append(adds.item())
    
    cls_add_dis[0].append(add.item())
    cls_adds_dis[0].append(adds.item())

    return (cls_add_dis, cls_adds_dis)


def eval_one_frame_pose_lm(
    item
):
    
    img_id, pcld, mask, ctr_of, pred_kp_of, gt_kp, gt_ctr, RTs, cls_ids, use_ctr, n_cls, \
        min_cnt, use_ctr_clus_flter, label, epoch, ibs, obj_id = item
    pred_pose_lst, pred_kp, gt_kp, gt_ctr = cal_frame_poses_lm(
        pcld, mask, ctr_of, pred_kp_of, gt_kp, gt_ctr, use_ctr, n_cls, use_ctr_clus_flter,
        obj_id
    )

    cls_add_dis, cls_adds_dis = eval_metric_lm(
         img_id, cls_ids, pred_pose_lst, RTs, mask, label, obj_id
    )
    return (cls_add_dis, cls_adds_dis, pred_kp, gt_kp, gt_ctr)

def eval_one_frame_pose_lab(
    item
):
      
    
    pcld, mask, pred_ctr_of, pred_kp_of, cls_id, use_ctr, \
    n_cls, min_cnt_lst, use_ctr_clus_flter, \
    epoch_lst, bs_lst, obj_id = item
        
        
    pred_pose = cal_frame_poses_lab(
        cls_id, pcld, mask, pred_ctr_of, pred_kp_of, use_ctr, n_cls, use_ctr_clus_flter,
        obj_id
    )

    
    return pred_pose
def eval_one_frame_pose_lm_icp(
    item
):
    
    img_id, pcld, mask, ctr_of, pred_kp_of, gt_kp, gt_ctr, RTs, cls_ids, use_ctr, n_cls, \
        min_cnt, use_ctr_clus_flter, label, epoch, ibs, obj_id = item
    pred_pose_lst, pred_kp, gt_kp, gt_ctr = cal_frame_poses_lm(
        pcld, mask, ctr_of, pred_kp_of, gt_kp, gt_ctr, use_ctr, n_cls, use_ctr_clus_flter,
        obj_id
    )

    cls_add_dis, cls_adds_dis = eval_metric_lm_icp(
         img_id, cls_ids, pred_pose_lst, RTs, mask, label, obj_id
    )
    return (cls_add_dis, cls_adds_dis, pred_kp, gt_kp, gt_ctr)
def vis_one_frame_pose_lm(
    item
):
    
    img_id, pcld, mask, ctr_of, pred_kp_of, gt_kp, gt_ctr, RTs, cls_ids, use_ctr, n_cls, \
        min_cnt, use_ctr_clus_flter, label, epoch, ibs, obj_id = item
    pred_pose_lst, pred_kp, gt_kp, gt_ctr = cal_frame_poses_lm(
        pcld, mask, ctr_of, pred_kp_of, gt_kp, gt_ctr, use_ctr, n_cls, use_ctr_clus_flter,
        obj_id
    )

    pred_RTs, gt_RTs = eval_metric_lm_vis(
         img_id, cls_ids, pred_pose_lst, RTs, mask, label, obj_id, pred_kp, gt_kp, gt_ctr
    )
    return (pred_RTs, gt_RTs, pred_kp, gt_kp, gt_ctr)
def vis_one_frame_pose_occlm(
    item
):
    
    img_id, pcld, mask, ctr_of, pred_kp_of, gt_kp, gt_ctr, RTs, cls_ids, use_ctr, n_cls, \
        min_cnt, use_ctr_clus_flter, label, epoch, ibs, obj_id = item
    pred_pose_lst, pred_kp, gt_kp, gt_ctr = cal_frame_poses_occlm(
        pcld, mask, ctr_of, pred_kp_of, gt_kp, gt_ctr, use_ctr, n_cls, use_ctr_clus_flter,
        obj_id
    )

    pred_RTs, gt_RTs = eval_metric_occlm_vis(
        img_id, cls_ids, pred_pose_lst, RTs, mask, label, obj_id
    )
    return (pred_RTs, gt_RTs, pred_kp, gt_kp, gt_ctr)
def eval_one_frame_pose_occlm(
    item
):
    
    img_id, pcld, mask, ctr_of, pred_kp_of, gt_kp, gt_ctr, RTs, cls_ids, use_ctr, n_cls, \
        min_cnt, use_ctr_clus_flter, label, epoch, ibs, obj_id = item
    pred_pose_lst, pred_kp, gt_kp, gt_ctr = cal_frame_poses_occlm(
        pcld, mask, ctr_of, pred_kp_of, gt_kp, gt_ctr, use_ctr, n_cls, use_ctr_clus_flter,
        obj_id
    )

    cls_add_dis, cls_adds_dis = eval_metric_occlm(
        img_id, cls_ids, pred_pose_lst, RTs, mask, label, obj_id
    )
    return (cls_add_dis, cls_adds_dis, pred_kp, gt_kp, gt_ctr)
# ###############################End LineMOD Evaluation###############################


# ###############################Shared Evaluation Entry###############################
class TorchEval():

    def __init__(self):
        
        if opt.dataset_name=='linemod':
            n_cls=2
            self.n_cls=2
        elif opt.dataset_name=="ycb":
            n_cls = 22
            self.n_cls = 22
        else:
            n_cls=2
            self.n_cls=2
        self.cls_add_dis = [list() for i in range(n_cls)]
        self.cls_adds_dis = [list() for i in range(n_cls)]
        self.cls_add_s_dis = [list() for i in range(n_cls)]
        self.pred_kp_errs = [list() for i in range(n_cls)]
        self.pred_id2pose_lst = []
        self.sym_cls_ids = []

    def cal_auc(self):
        add_auc_lst = []
        adds_auc_lst = []
        add_s_auc_lst = []
        for cls_id in range(1, self.n_cls):
            if (cls_id) in config.ycb_sym_cls_ids:
                self.cls_add_s_dis[cls_id] = self.cls_adds_dis[cls_id]
            else:
                self.cls_add_s_dis[cls_id] = self.cls_add_dis[cls_id]
            self.cls_add_s_dis[0] += self.cls_add_s_dis[cls_id]
        for i in range(self.n_cls):
            add_auc = bs_utils.cal_auc(self.cls_add_dis[i])
            adds_auc = bs_utils.cal_auc(self.cls_adds_dis[i])
            add_s_auc = bs_utils.cal_auc(self.cls_add_s_dis[i])
            add_auc_lst.append(add_auc)
            adds_auc_lst.append(adds_auc)
            add_s_auc_lst.append(add_s_auc)
            if i == 0:
                continue
            print(cls_lst[i-1])
            print("***************add:\t", add_auc)
            print("***************adds:\t", adds_auc)
            print("***************add(-s):\t", add_s_auc)
        # kp errs:
        n_objs = sum([len(l) for l in self.pred_kp_errs])
        all_errs = 0.0
        for cls_id in range(1, self.n_cls):
            all_errs += sum(self.pred_kp_errs[cls_id])
        print("mean kps errs:", all_errs / n_objs)

        print("Average of all object:")
        print("***************add:\t", np.mean(add_auc_lst[1:]))
        print("***************adds:\t", np.mean(adds_auc_lst[1:]))
        print("***************add(-s):\t", np.mean(add_s_auc_lst[1:]))

        print("All object (following PoseCNN):")
        print("***************add:\t", add_auc_lst[0])
        print("***************adds:\t", adds_auc_lst[0])
        print("***************add(-s):\t", add_s_auc_lst[0])

        sv_info = dict(
            add_dis_lst=self.cls_add_dis,
            adds_dis_lst=self.cls_adds_dis,
            add_auc_lst=add_auc_lst,
            adds_auc_lst=adds_auc_lst,
            add_s_auc_lst=add_s_auc_lst,
            pred_kp_errs=self.pred_kp_errs,
        )
        sv_pth = os.path.join(
            opt.log_eval_dir,
            'pvn3d_eval_cuda_{}_{}_{}.pkl'.format(
                adds_auc_lst[0], add_auc_lst[0], add_s_auc_lst[0]
            )
        )
        pkl.dump(sv_info, open(sv_pth, 'wb'))
        sv_pth = os.path.join(
            opt.log_eval_dir,
            'pvn3d_eval_cuda_{}_{}_{}_id2pose.pkl'.format(
                adds_auc_lst[0], add_auc_lst[0], add_s_auc_lst[0]
            )
        )
        pkl.dump(self.pred_id2pose_lst, open(sv_pth, 'wb'))
    def cal_lmo_add(self, obj_id, test_occ=False):
        
        add_auc_lst = []
        adds_auc_lst = []
        add_s_auc_lst = []
        # cls_id = obj_id
        cls_id = 0
        if (obj_id) in config_lmo.lm_sym_cls_ids:
            self.cls_add_s_dis[cls_id] = self.cls_adds_dis[cls_id]
        else:
            self.cls_add_s_dis[cls_id] = self.cls_add_dis[cls_id]
        self.cls_add_s_dis[0] += self.cls_add_s_dis[cls_id]
        add_auc = bs_utils_lmo.cal_auc(self.cls_add_dis[cls_id])
        adds_auc = bs_utils_lmo.cal_auc(self.cls_adds_dis[cls_id])
        add_s_auc = bs_utils_lmo.cal_auc(self.cls_add_s_dis[cls_id])
        add_auc_lst.append(add_auc)
        adds_auc_lst.append(adds_auc)
        add_s_auc_lst.append(add_s_auc)
        d = config_lmo.lm_r_lst[obj_id]['diameter'] / 1000.0 * 0.1
        print("obj_id: ", obj_id, "0.1 diameter: ", d)
        add = np.mean(np.array(self.cls_add_dis[cls_id]) < d) * 100
        adds = np.mean(np.array(self.cls_adds_dis[cls_id]) < d) * 100

        cls_type = config_lmo.lmo_id2obj_dict[obj_id]
        print(obj_id, cls_type)
        print("***************add auc:\t", add_auc)
        print("***************adds auc:\t", adds_auc)
        print("***************add(-s) auc:\t", add_s_auc)
        print("***************add < 0.1 diameter:\t", add)
        print("***************adds < 0.1 diameter:\t", adds)

        sv_info = dict(
            add_dis_lst=self.cls_add_dis,
            adds_dis_lst=self.cls_adds_dis,
            add_auc_lst=add_auc_lst,
            adds_auc_lst=adds_auc_lst,
            add_s_auc_lst=add_s_auc_lst,
            add=add,
            adds=adds,
        )
        occ = "occlusion" if test_occ else ""
        sv_pth = os.path.join(
            opt.log_eval_dir,
            'pvn3d_eval_cuda_{}_{}_{}_{}.pkl'.format(
                cls_type, occ, add, adds
            )
        )
        pkl.dump(sv_info, open(sv_pth, 'wb'))
    def cal_lm_add(self, obj_id, test_occ=False):
        
        add_auc_lst = []
        adds_auc_lst = []
        add_s_auc_lst = []
        # cls_id = obj_id
        cls_id = 0
        if (obj_id) in config_lm.lm_sym_cls_ids:
            self.cls_add_s_dis[cls_id] = self.cls_adds_dis[cls_id]
        else:
            self.cls_add_s_dis[cls_id] = self.cls_add_dis[cls_id]
        self.cls_add_s_dis[0] += self.cls_add_s_dis[cls_id]
        add_auc = bs_utils_lm.cal_auc(self.cls_add_dis[cls_id])
        adds_auc = bs_utils_lm.cal_auc(self.cls_adds_dis[cls_id])
        add_s_auc = bs_utils_lm.cal_auc(self.cls_add_s_dis[cls_id])
        add_auc_lst.append(add_auc)
        adds_auc_lst.append(adds_auc)
        add_s_auc_lst.append(add_s_auc)
        d = config_lm.lm_r_lst[obj_id]['diameter'] / 1000.0 * 0.1
        print("obj_id: ", obj_id, "0.1 diameter: ", d)
        add = np.mean(np.array(self.cls_add_dis[cls_id]) < d) * 100
        adds = np.mean(np.array(self.cls_adds_dis[cls_id]) < d) * 100

        cls_type = config_lm.lm_id2obj_dict[obj_id]
        print(obj_id, cls_type)
        print("***************add auc:\t", add_auc)
        print("***************adds auc:\t", adds_auc)
        print("***************add(-s) auc:\t", add_s_auc)
        print("***************add < 0.1 diameter:\t", add)
        print("***************adds < 0.1 diameter:\t", adds)

        sv_info = dict(
            add_dis_lst=self.cls_add_dis,
            adds_dis_lst=self.cls_adds_dis,
            add_auc_lst=add_auc_lst,
            adds_auc_lst=adds_auc_lst,
            add_s_auc_lst=add_s_auc_lst,
            add=add,
            adds=adds,
        )
        occ = "occlusion" if test_occ else ""
        sv_pth = os.path.join(
            opt.log_eval_dir,
            'pvn3d_eval_cuda_{}_{}_{}_{}.pkl'.format(
                cls_type, occ, add, adds
            )
        )
        pkl.dump(sv_info, open(sv_pth, 'wb'))
        
    def eval_pose_parallel_icp(
            self, pclds, img_id, rgbs, masks, pred_ctr_ofs, gt_ctr_ofs, labels, cnt,
            cls_ids, RTs, pred_kp_ofs, gt_kps, gt_ctrs, min_cnt=20, merge_clus=False,
            use_ctr_clus_flter=True, use_ctr=True, obj_id=0, kp_type='farthest',
            ds='ycb'
        ):
            
            bs, n_kps, n_pts, c = pred_kp_ofs.size()
            masks = masks.long()
            cls_ids = cls_ids.long()
            use_ctr_lst = [use_ctr for i in range(bs)]
            n_cls_lst = [self.n_cls for i in range(bs)]
            min_cnt_lst = [min_cnt for i in range(bs)]
            epoch_lst = [cnt*bs for i in range(bs)]
            bs_lst = [i for i in range(bs)]
            use_ctr_clus_flter_lst = [use_ctr_clus_flter for i in range(bs)]
            obj_id_lst = [obj_id for i in range(bs)]
            kp_type = [kp_type for i in range(bs)]
            if ds == "ycb":
                data_gen = zip(
                    pclds, masks, pred_ctr_ofs, pred_kp_ofs, RTs,
                    cls_ids, use_ctr_lst, n_cls_lst, min_cnt_lst, use_ctr_clus_flter_lst,
                    labels, epoch_lst, bs_lst, gt_kps, gt_ctrs, kp_type
                )
            elif ds=="linemod":
                data_gen = zip(
                    img_id, pclds, masks, pred_ctr_ofs, pred_kp_ofs, gt_kps, gt_ctrs, RTs,
                    cls_ids, use_ctr_lst, n_cls_lst, min_cnt_lst, use_ctr_clus_flter_lst,
                    labels, epoch_lst, bs_lst, obj_id_lst
                )
            else:
                data_gen = zip(
                    pclds, masks, pred_ctr_ofs, pred_kp_ofs, gt_kps, gt_ctrs, RTs,
                    cls_ids, use_ctr_lst, n_cls_lst, min_cnt_lst, use_ctr_clus_flter_lst,
                    labels, epoch_lst, bs_lst, obj_id_lst
                )
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=bs
            ) as executor:
            
                if ds == "ycb":
                    eval_func = eval_one_frame_pose
                elif ds == "linemod":
                    eval_func = eval_one_frame_pose_lm_icp
                else:
                    eval_func = eval_one_frame_pose_occlm
                for res in executor.map(eval_func, data_gen):
                    if ds == 'ycb':
                        cls_add_dis_lst, cls_adds_dis_lst, pred_cls_ids, pred_poses, pred_kp_errs = res
                        self.pred_id2pose_lst.append(
                            {cid: pose for cid, pose in zip(pred_cls_ids, pred_poses)}
                        )
                        self.pred_kp_errs = self.merge_lst(
                            self.pred_kp_errs, pred_kp_errs
                        )
                    elif ds == "linemod":
                        cls_add_dis_lst, cls_adds_dis_lst, pred_kp, gt_kp, gt_ctr = res
                    else:
                        
                        cls_add_dis_lst, cls_adds_dis_lst, pred_kp, gt_kp, gt_ctr = res
                    
                    self.cls_add_dis = self.merge_lst(
                        self.cls_add_dis, cls_add_dis_lst
                    )
                    self.cls_adds_dis = self.merge_lst(
                        self.cls_adds_dis, cls_adds_dis_lst
                    )
            
            return (cls_add_dis_lst[0][0], cls_adds_dis_lst[0][0], pred_kp, gt_kp, gt_ctr)    
    def eval_pose_parallel_vis(
        self, pclds, img_id, rgbs, masks, pred_ctr_ofs, gt_ctr_ofs, labels, cnt,
        cls_ids, RTs, pred_kp_ofs, gt_kps, gt_ctrs, min_cnt=20, merge_clus=False,
        use_ctr_clus_flter=True, use_ctr=True, obj_id=0, kp_type='farthest',
        ds='ycb'
    ):
        
        bs, n_kps, n_pts, c = pred_kp_ofs.size()
        masks = masks.long()
        cls_ids = cls_ids.long()
        use_ctr_lst = [use_ctr for i in range(bs)]
        n_cls_lst = [self.n_cls for i in range(bs)]
        min_cnt_lst = [min_cnt for i in range(bs)]
        epoch_lst = [cnt*bs for i in range(bs)]
        bs_lst = [i for i in range(bs)]
        use_ctr_clus_flter_lst = [use_ctr_clus_flter for i in range(bs)]
        obj_id_lst = [obj_id for i in range(bs)]
        kp_type = [kp_type for i in range(bs)]
        if ds == "ycb":
            data_gen = zip(
                pclds, masks, pred_ctr_ofs, pred_kp_ofs, RTs,
                cls_ids, use_ctr_lst, n_cls_lst, min_cnt_lst, use_ctr_clus_flter_lst,
                labels, epoch_lst, bs_lst, gt_kps, gt_ctrs, kp_type
            )
        elif ds=="linemod":
            data_gen = zip(
                img_id, pclds, masks, pred_ctr_ofs, pred_kp_ofs, gt_kps, gt_ctrs, RTs,
                cls_ids, use_ctr_lst, n_cls_lst, min_cnt_lst, use_ctr_clus_flter_lst,
                labels, epoch_lst, bs_lst, obj_id_lst
            )
        else:
            data_gen = zip(
                img_id, pclds, masks, pred_ctr_ofs, pred_kp_ofs, gt_kps, gt_ctrs, RTs,
                cls_ids, use_ctr_lst, n_cls_lst, min_cnt_lst, use_ctr_clus_flter_lst,
                labels, epoch_lst, bs_lst, obj_id_lst
            )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=bs
        ) as executor:
        
            if ds == "ycb":
                eval_func = eval_one_frame_pose
            elif ds == "linemod":
                eval_func = vis_one_frame_pose_lm
            else:
                eval_func = vis_one_frame_pose_occlm
            for res in executor.map(eval_func, data_gen):
                if ds == 'ycb':
                    cls_add_dis_lst, cls_adds_dis_lst, pred_cls_ids, pred_poses, pred_kp_errs = res
                    self.pred_id2pose_lst.append(
                        {cid: pose for cid, pose in zip(pred_cls_ids, pred_poses)}
                    )
                    self.pred_kp_errs = self.merge_lst(
                        self.pred_kp_errs, pred_kp_errs
                    )
                elif ds == "linemod":
                    pred_RTs, gt_RTs, pred_kp, gt_kp, gt_ctr = res
                else:
                    
                    pred_RTs, gt_RTs, pred_kp, gt_kp, gt_ctr = res
                
                # self.cls_add_dis = self.merge_lst(
                #     self.cls_add_dis, cls_add_dis_lst
                # )
                # self.cls_adds_dis = self.merge_lst(
                #     self.cls_adds_dis, cls_adds_dis_lst
                # )
        
        return (pred_RTs, gt_RTs, pred_kp, gt_kp, gt_ctr)
    def eval_pose_parallel_lab(
        self, pclds, masks, pred_ctr_ofs, cnt,
        cls_ids, pred_kp_ofs, min_cnt=20, merge_clus=False,
        use_ctr_clus_flter=True, use_ctr=True, obj_id=0, kp_type='farthest',
        ds='lab'
    ):
        
        bs, n_kps, n_pts, c = pred_kp_ofs.size()
        masks = masks.long()
        cls_ids = [cls_ids]
        use_ctr_lst = [use_ctr for i in range(bs)]
        n_cls_lst = [self.n_cls for i in range(bs)]
        min_cnt_lst = [min_cnt for i in range(bs)]
        epoch_lst = [cnt*bs for i in range(bs)]
        bs_lst = [i for i in range(bs)]
        use_ctr_clus_flter_lst = [use_ctr_clus_flter for i in range(bs)]
        obj_id_lst = [obj_id for i in range(bs)]
        kp_type = [kp_type for i in range(bs)]
        
        data_gen = zip(
            pclds, masks, pred_ctr_ofs, pred_kp_ofs,
            cls_ids, use_ctr_lst, n_cls_lst, min_cnt_lst, use_ctr_clus_flter_lst,
            epoch_lst, bs_lst, obj_id_lst
        )
        
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=bs
        ) as executor:
            eval_func = eval_one_frame_pose_lab
            
            for res in executor.map(eval_func, data_gen):
                pred_pose = res
                
        return pred_pose

    def eval_pose_parallel(
        self, pclds, img_id, rgbs, masks, pred_ctr_ofs, gt_ctr_ofs, labels, cnt,
        cls_ids, RTs, pred_kp_ofs, gt_kps, gt_ctrs, min_cnt=20, merge_clus=False,
        use_ctr_clus_flter=True, use_ctr=True, obj_id=0, kp_type='farthest',
        ds='ycb'
    ):
        
        bs, n_kps, n_pts, c = pred_kp_ofs.size()
        masks = masks.long()
        cls_ids = cls_ids.long()
        use_ctr_lst = [use_ctr for i in range(bs)]
        n_cls_lst = [self.n_cls for i in range(bs)]
        min_cnt_lst = [min_cnt for i in range(bs)]
        epoch_lst = [cnt*bs for i in range(bs)]
        bs_lst = [i for i in range(bs)]
        use_ctr_clus_flter_lst = [use_ctr_clus_flter for i in range(bs)]
        obj_id_lst = [obj_id for i in range(bs)]
        kp_type = [kp_type for i in range(bs)]
        if ds == "ycb":
            data_gen = zip(
                pclds, masks, pred_ctr_ofs, pred_kp_ofs, RTs,
                cls_ids, use_ctr_lst, n_cls_lst, min_cnt_lst, use_ctr_clus_flter_lst,
                labels, epoch_lst, bs_lst, gt_kps, gt_ctrs, kp_type
            )
        elif ds=="linemod":
            data_gen = zip(
                img_id, pclds, masks, pred_ctr_ofs, pred_kp_ofs, gt_kps, gt_ctrs, RTs,
                cls_ids, use_ctr_lst, n_cls_lst, min_cnt_lst, use_ctr_clus_flter_lst,
                labels, epoch_lst, bs_lst, obj_id_lst
            )
        else:
            data_gen = zip(
                pclds, masks, pred_ctr_ofs, pred_kp_ofs, gt_kps, gt_ctrs, RTs,
                cls_ids, use_ctr_lst, n_cls_lst, min_cnt_lst, use_ctr_clus_flter_lst,
                labels, epoch_lst, bs_lst, obj_id_lst
            )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=bs
        ) as executor:
        
            if ds == "ycb":
                eval_func = eval_one_frame_pose
            elif ds == "linemod":
                eval_func = eval_one_frame_pose_lm
            else:
                eval_func = eval_one_frame_pose_occlm
            for res in executor.map(eval_func, data_gen):
                if ds == 'ycb':
                    cls_add_dis_lst, cls_adds_dis_lst, pred_cls_ids, pred_poses, pred_kp_errs = res
                    self.pred_id2pose_lst.append(
                        {cid: pose for cid, pose in zip(pred_cls_ids, pred_poses)}
                    )
                    self.pred_kp_errs = self.merge_lst(
                        self.pred_kp_errs, pred_kp_errs
                    )
                elif ds == "linemod":
                    cls_add_dis_lst, cls_adds_dis_lst, pred_kp, gt_kp, gt_ctr = res
                else:
                    
                    cls_add_dis_lst, cls_adds_dis_lst, pred_kp, gt_kp, gt_ctr = res
                
                self.cls_add_dis = self.merge_lst(
                    self.cls_add_dis, cls_add_dis_lst
                )
                self.cls_adds_dis = self.merge_lst(
                    self.cls_adds_dis, cls_adds_dis_lst
                )
        
        return (cls_add_dis_lst[0][0], cls_adds_dis_lst[0][0], pred_kp, gt_kp, gt_ctr)

    def merge_lst(self, targ, src):
        
        for i in range(len(targ)):
            targ[i] += src[i]
        return targ

# vim: ts=4 sw=4 sts=4 expandtab