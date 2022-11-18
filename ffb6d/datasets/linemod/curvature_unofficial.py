# calculate curavature
import os
import h5py
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from PIL import Image
import os
import yaml
import normalSpeed
import tqdm

epsilon = 1E-10
def get_KNN_points(center_point, xyz, k):
    new = np.tile(center_point, (xyz.shape[0], 1))
    delta = xyz - new
    dist = np.sum(delta * delta, 1)
    dist = torch.from_numpy(dist)
    sorteDis, pos = dist.sort()
    knn_points_ids = pos[1:k + 1]
    knn_points = xyz[knn_points_ids, :]


    return knn_points


def get_triangle_angles(center_point, p1, p2):
    p0p1 = math.sqrt((center_point[0] - p1[0]) ** 2 + (center_point[1] - p1[1]) ** 2 + (center_point[2] - p1[2]) ** 2)
    p1p2 = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)
    p0p2 = math.sqrt((center_point[0] - p2[0]) ** 2 + (center_point[1] - p2[1]) ** 2 + (center_point[2] - p2[2]) ** 2)
    a = (p0p1 ** 2 + p1p2 ** 2 - p0p2 ** 2) / 2 / p0p1 / p1p2
    b = (p0p2 ** 2 + p1p2 ** 2 - p0p1 ** 2) / 2 / p0p2 / p1p2
    # ensure not get out the bound of [-1,1]
    if a<-1:
        a = -0.9999999
    if a>1:
        a = 0.9999999
    if b<-1:
        b = -0.9999999
    if b>1:
        b = 0.9999999
        
    angleP1 = math.acos(a) * 180.0 / math.pi
    angleP2 = math.acos(b) * 180.0 / math.pi
    angleP0 = 180 - angleP1 - angleP2
    return angleP0, angleP1, angleP2, p1p2, p0p2, p0p1


def get_triangle_area( angleP0, angleP1, angleP2, p1p2, p0p2, p0p1 ):
    R = p1p2 / (2 * math.sin((angleP0 / 180) * math.pi))
    if angleP0 < 90:
        A = R * R - p0p1 * p0p1 / 4
        B = R * R - p0p2 * p0p2 / 4
        if epsilon > abs(A):
            A = 0
        if epsilon > abs(B):
            B = 0
            
        Area = (math.sqrt(A) * p0p1) / 4 + (math.sqrt(B) * p0p1) / 4

    else:
        Area = ((p1p2 * p0p1 * p0p2) / (4 * R)) / 2

    return Area



def get_curcature(center_point, center_normal, knn_points):
    KH = 0
    KG = 0
    # k = size(knn_points, 2)
    Area = np.zeros((k, 1))
    angles_P0 = np.zeros((k, 1))
    angles_P1 = np.zeros((k, 1))
    angles_P2 = np.zeros((k, 1))

    # get angles and areas

    for m in range(k):

        p1 = knn_points[m, :]

        if (m == int(k - 1)):

            p2 = knn_points[1, :]
        else:
            p2 = knn_points[m + 1, :]

        # get angels of triangle
        angleP0, angleP1, angleP2, p1p2, p0p2, p0p1 = get_triangle_angles(center_point, p1, p2)
        angles_P0[m, :] = angleP0 * math.pi / 180
        angles_P1[m, :] = angleP1 * math.pi / 180
        angles_P2[m, :] = angleP2 * math.pi / 180
        # get Area of triangle
        if angleP0 == 0:
            angleP0 = 0.000001
            print(angleP0)
        Area[m, :] = get_triangle_area(angleP0, angleP1, angleP2, p1p2, p0p2, p0p1)

    # cal curcature
    for m in range(k):

        if m == 1:
            aplha = angles_P1[k-1,:]
            beta = angles_P2[m,:]
        else:

            aplha = angles_P1[m - 1,:]
            beta = angles_P2[m,:]
        P1P0 = knn_points[m,:] - center_point
        try:
            KH = KH + (1/math.tan(aplha) + 1/math.tan(beta) ) * (sum(P1P0 * center_normal))
        except ZeroDivisionError:
            KH = KH
        KG = KG + angles_P0[m,:]


    Am = sum(Area)
    KH = KH / (4 * Am)
    KG = (2 * math.pi - KG) / Am

    K1 = KH + (KH ** 2 - KG) ** (1 / 2)
    K2 = KH - (KH ** 2 - KG) ** (1 / 2)

    return KH,KG,K1,K2


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data']
    n = f['normal']
    return (data, n)

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

K=np.array([[572.4114, 0.,         325.2611],
            [0.,        573.57043,  242.04899],
            [0.,        0.,         1.]])
meta_file = open(os.path.join('/workspace/DATA/Linemod_preprocessed/data','15', 'gt.yml'), "r")
meta_lst = yaml.safe_load(meta_file)

# for item_name in tqdm.tqdm(trainlist):
item_name = '0006'
with Image.open(os.path.join('/workspace/DATA/Linemod_preprocessed/data','15', "depth/{}.png".format(item_name))) as di:
    dpt_mm = np.array(di)

meta = meta_lst[int(item_name)]
meta = meta[0]

cam_scale = 1000.0
#dpt_mm = fill_missing(dpt_mm, cam_scale, 1)

dpt_mm = dpt_mm.copy().astype(np.uint16)
#nrm_map = normalSpeed.depth_normal(dpt_mm, K[0][0], K[1][1], 5, 2000, 20, False)

# if True:
#     show_nrm_map = ((nrm_map + 1.0) * 127).astype(np.uint8)
#     img_file = os.path.join('/workspace/DATA/Linemod_preprocessed/data',args.cls_num,'vis_nrm_{}.png'.format(item_name))
#     cv2.imwrite(img_file, show_nrm_map)
dpt_m = dpt_mm.astype(np.float32) / cam_scale
dpt_xyz = dpt_2_pcld(dpt_m, 1.0, K)
dpt_xyz[np.isnan(dpt_xyz)] = 0.0
dpt_xyz[np.isinf(dpt_xyz)] = 0.0

nrm_map = normalSpeed.depth_normal(dpt_mm, K[0][0], K[1][1], 5, 2000, 20, False)
#xyz1, normal1 = load_h5('/workspace/modelnet40_ply_hdf5_2048/ply_data_test1.h5')
#  you can download the dataset from 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'

k = 7
msk_dp = dpt_mm > 1e-6
choose = msk_dp.flatten().nonzero()[0].astype(np.uint32)
choose_2 = np.array([i for i in range(len(choose))])

if len(choose_2) > 480*640//24:
    c_mask = np.zeros(len(choose_2), dtype=int)
    c_mask[:480*640//24] = 1
    np.random.shuffle(c_mask)
    choose_2 = choose_2[c_mask.nonzero()]
else:
    choose_2 = np.pad(choose_2, (0, 480*640//24-len(choose_2)), 'wrap')
choose = np.array(choose)[choose_2]

sf_idx = np.arange(choose.shape[0])
np.random.shuffle(sf_idx)
choose = choose[sf_idx]

cld = dpt_xyz.reshape(-1, 3)[choose, :]

nrm_pt = nrm_map[:, :, :3].reshape(-1, 3)[choose, :]
KH = np.zeros((len(choose), 1))
KG = np.zeros((len(choose), 1))
K1 = np.zeros((len(choose), 1))
K2 = np.zeros((len(choose), 1))
for i in tqdm.tqdm(range(len(cld))):
    
    if sum(cld[i])==0 or sum(nrm_pt[i])==0:
        KH[i]=0
        KG[i]=0
        K1[i]=0
        K2[i]=0
        continue
    else:
        
        xyz = cld[i]
        normal = nrm_pt[i]
        center_point = xyz
        center_normal = normal
        knn_points = get_KNN_points(center_point, cld, k)
        KH[i], KG[i], K1[i], K2[i] = get_curcature(center_point, center_normal, knn_points)

cur_file = os.path.join('/workspace/DATA/Linemod_preprocessed/data','15','cur_{}'.format(item_name))
np.savez_compressed(cur_file, KH=KH, KG=KG, K1=K1,K2=K2,cld=cld)   
    # where K1,k2 are the Principal curvature, KH is Mean curvature， KG is Gaussian curvature

cur = np.load('/workspace/DATA/Linemod_preprocessed/data/15/cur_0006.npz')
cld_file = '/workspace/DATA/Linemod_preprocessed/data/15/cur_0006_cld.txt'
KH_file = '/workspace/DATA/Linemod_preprocessed/data/15/cur_0006_KH.txt'
KG_file = '/workspace/DATA/Linemod_preprocessed/data/15/cur_0006_KG.txt'
K1_file = '/workspace/DATA/Linemod_preprocessed/data/15/cur_0006_K1.txt'
K2_file = '/workspace/DATA/Linemod_preprocessed/data/15/cur_0006_K2.txt'
np.savetxt(cld_file,cur['cld'])
np.savetxt(KH_file,cur['KH'])
np.savetxt(KG_file,cur['KG'])
np.savetxt(K1_file,cur['K1'])
np.savetxt(K2_file,cur['K2'])