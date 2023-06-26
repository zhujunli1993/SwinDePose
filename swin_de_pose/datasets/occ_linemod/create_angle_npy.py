import numpy as np
import math
from PIL import Image
import yaml
import os
import cv2
import tqdm
import normalSpeed
import json
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pdb
import depth_map_utils_ycb as depth_map_utils

# def curv(points):
#     neighborhood_size=50
#     points = torch.from_numpy(points).float()
#     points = points.reshape(1,480*640,3)
#     curvatures,_ = estimate_pointcloud_local_coord_frames(points,neighborhood_size=neighborhood_size)

parser = ArgumentParser()

# parser.add_argument(
#     '--train_list', type=str, 
#     help="training list for real/ generation."
# )
# parser.add_argument(
#     '--test_list', type=str, 
#     help="testing list for real/ generation."
# )
parser.add_argument(
    '--vis_img', action="store_true",
    help="visulaize histogram."
)
args = parser.parse_args()
def fill_missing(
            dpt, cam_scale, scale_2_80m, fill_type='multiscale',
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

def pseudo_nrm_angle(nrm_map):
    height=480
    width=640
    # Set up-axis 
    x_up = np.array([1.0, 0.0, 0.0])
    y_up = np.array([0.0, 1.0, 0.0])
    z_up = np.array([0.0, 0.0, 1.0])

    angle_x = []
    # signed_x = []
    angle_y = []
    # signed_y = []
    angle_z = []
    # signed_z = []
    # dpt_xyz = np.reshape(dpt_xyz,(height, width, 3))
    for i in range(0, height):
        for j in range(0, width):
            if sum(nrm_map[i, j])==0.0:
                angle_x.append(360.0)
                angle_y.append(360.0)
                angle_z.append(360.0)
                continue
            else:
                nrm_c = nrm_map[i, j]
                epsilon = 1E-6
                value_x = nrm_c[0]*x_up[0] + nrm_c[1]*x_up[1] + nrm_c[2]*x_up[2]
                value_y = nrm_c[0]*y_up[0] + nrm_c[1]*y_up[1] + nrm_c[2]*y_up[2]
                value_z = nrm_c[0]*z_up[0] + nrm_c[1]*z_up[1] + nrm_c[2]*z_up[2]
                if value_x > (1.0 - epsilon):
                    value_x = 1.0
                elif value_x < (epsilon - 1.0):
                    value_x = -1.0 
                angle_x.append(np.arccos(value_x)*180.0/math.pi)
                
                if value_y > (1.0 - epsilon):
                    value_y = 1.0
                elif value_y < (epsilon - 1.0):
                    value_y = -1.0 
                angle_y.append(np.arccos(value_y)*180.0/math.pi)
                
                if value_z > (1.0 - epsilon):
                    value_z = 1.0
                elif value_z < (epsilon - 1.0):
                    value_z = -1.0 
                angle_z.append(np.arccos(value_z)*180.0/math.pi)
    angle_x = np.reshape(angle_x, [height, width])
    # signed_x = np.reshape(signed_x, [height, width])
    angle_y = np.reshape(angle_y, [height, width])
    # signed_y = np.reshape(signed_y, [height, width])
    angle_z = np.reshape(angle_z, [height, width])            
    if args.vis_img:
        angle_x[angle_x==360] = 255
        angle_x = (angle_x-angle_x[angle_x<255].min())*(254/(angle_x[angle_x<255].max()-angle_x[angle_x<255].min()))
        angle_y[angle_y==360] = 255
        angle_y = (angle_y-angle_y[angle_y<255].min())*(254/(angle_y[angle_y<255].max()-angle_y[angle_y<255].min()))
        angle_z[angle_z==360] = 255
        angle_z = (angle_z-angle_z[angle_z<255].min())*(254/(angle_z[angle_z<255].max()-angle_z[angle_z<255].min()))
        # combine three channels and save to a png image
        new_img_angles = np.dstack((angle_x, angle_y))
        new_img_angles = np.dstack((new_img_angles, angle_z))
        new_img_angles = new_img_angles.astype(np.uint8)
    else:
        new_img_angles = np.dstack((angle_x, angle_y))
        new_img_angles = np.dstack((new_img_angles, angle_z))
    return new_img_angles     

       
def pseudo_gen(args, dpt_xyz):
    height=480
    width=640
    # Set up-axis 
    x_up = np.array([1.0, 0.0, 0.0])
    y_up = np.array([0.0, 1.0, 0.0])
    z_up = np.array([0.0, 0.0, 1.0])

    angle_x = []
    # signed_x = []
    angle_y = []
    # signed_y = []
    angle_z = []
    # signed_z = []
    dpt_xyz = np.reshape(dpt_xyz,(height, width, 3))
    for i in range(0, height):
        for j in range(0, width):
            p_nei = []
            p_c = dpt_xyz[i, j]
            if sum(p_c)==0.0:
                angle_x.append(360.0)
                angle_y.append(360.0)
                angle_z.append(360.0)
                # signed_x.append(360.0)
                # signed_y.append(360.0)
                # signed_z.append(360.0)
                continue
            
            else:
                if i == 0 and j == 0:
                    p_nei.append(dpt_xyz[i, j+1])
                    p_nei.append(dpt_xyz[i+1, j])
                    p_nei.append(dpt_xyz[i+1, j+1])
                if i == height-1 and j == width-1:
                    p_nei.append(dpt_xyz[i, j-1])
                    p_nei.append(dpt_xyz[i-1, j-1])
                    p_nei.append(dpt_xyz[i-1, j])
                if i == 0 and j > 0 and j < width-1:
                    p_nei.append(dpt_xyz[i, j-1])
                    p_nei.append(dpt_xyz[i, j+1])
                    p_nei.append(dpt_xyz[i+1, j-1])
                    p_nei.append(dpt_xyz[i+1, j])
                    p_nei.append(dpt_xyz[i+1, j+1])
                if i == 0 and j == width-1:
                    p_nei.append(dpt_xyz[i, j-1])
                    p_nei.append(dpt_xyz[i+1, j-1])
                    p_nei.append(dpt_xyz[i+1, j])
                if i > 0 and i < height-1 and j == 0:
                    p_nei.append(dpt_xyz[i-1, j])
                    p_nei.append(dpt_xyz[i-1, j+1])
                    p_nei.append(dpt_xyz[i, j+1])
                    p_nei.append(dpt_xyz[i+1, j])
                    p_nei.append(dpt_xyz[i+1, j+1])
                if i == height-1 and j == 0:
                    p_nei.append(dpt_xyz[i-1, j])
                    p_nei.append(dpt_xyz[i-1, j+1])
                    p_nei.append(dpt_xyz[i, j+1])   
                    
                if (i > 0 and i < height-1) and (j > 0 and j < width-1): 
                    p_nei.append(dpt_xyz[i-1, j-1])
                    p_nei.append(dpt_xyz[i-1, j])
                    p_nei.append(dpt_xyz[i-1, j+1])
                    p_nei.append(dpt_xyz[i, j-1])
                    p_nei.append(dpt_xyz[i, j+1])
                    p_nei.append(dpt_xyz[i+1, j-1])
                    p_nei.append(dpt_xyz[i+1, j])
                    p_nei.append(dpt_xyz[i+1, j+1])
                # p_2 = dpt_xyz[i, j]
                # p_1 = dpt_xyz[i, j-1]
                # if p_2[0]+p_2[1]+p_2[2]==0.0 or p_1[0]+p_1[1]+p_1[2]==0.0:
                
                epsilon = 1E-6
                #count = 0
                
                difference_nei = []
                for n in range(len(p_nei)):
                    if sum(p_nei[n])!=0:
                        difference = p_nei[n] - p_c
                        difference_lengh = np.sqrt(math.pow(difference[0],2)+math.pow(difference[1],2)+math.pow(difference[2],2))
                        if difference_lengh < epsilon:
                            continue
                        else:
                            difference_nei.append(difference)
                            #count+=1
                if len(difference_nei) == 0:
                    angle_x.append(360.0)
                    angle_y.append(360.0)
                    angle_z.append(360.0)
                    continue
                else:
                    
                    mean_diff = np.mean(difference_nei, axis=0)
                    mean_diff_len = np.sqrt(math.pow(mean_diff[0],2)+math.pow(mean_diff[1],2)+math.pow(mean_diff[2],2))
                    if mean_diff_len == 0:
                        angle_x.append(360.0)
                        angle_y.append(360.0)
                        angle_z.append(360.0)
                        continue    
                    else:
                        value_x = (mean_diff[0]*x_up[0] + mean_diff[1]*x_up[1] + mean_diff[2]*x_up[2]) / mean_diff_len
                        value_y = (mean_diff[0]*y_up[0] + mean_diff[1]*y_up[1] + mean_diff[2]*y_up[2]) / mean_diff_len
                        value_z = (mean_diff[0]*z_up[0] + mean_diff[1]*z_up[1] + mean_diff[2]*z_up[2]) / mean_diff_len
                        if value_x > (1.0 - epsilon):
                            value_x = 1.0
                        elif value_x < (epsilon - 1.0):
                            value_x = -1.0 
                        angle_x.append(np.arccos(value_x)*180.0/math.pi)
                        
                        if value_y > (1.0 - epsilon):
                            value_y = 1.0
                        elif value_y < (epsilon - 1.0):
                            value_y = -1.0 
                        angle_y.append(np.arccos(value_y)*180.0/math.pi)
                        
                        if value_z > (1.0 - epsilon):
                            value_z = 1.0
                        elif value_z < (epsilon - 1.0):
                            value_z = -1.0 
                        angle_z.append(np.arccos(value_z)*180.0/math.pi)
                
                
    angle_x = np.reshape(angle_x, [height, width])
    # signed_x = np.reshape(signed_x, [height, width])
    angle_y = np.reshape(angle_y, [height, width])
    # signed_y = np.reshape(signed_y, [height, width])
    angle_z = np.reshape(angle_z, [height, width])
    # signed_z = np.reshape(signed_z, [height, width])

    if args.vis_img:
        angle_x[angle_x==360] = 255
        angle_x = (angle_x-angle_x[angle_x<255].min())*(254/(angle_x[angle_x<255].max()-angle_x[angle_x<255].min()))
        angle_y[angle_y==360] = 255
        angle_y = (angle_y-angle_y[angle_y<255].min())*(254/(angle_y[angle_y<255].max()-angle_y[angle_y<255].min()))
        angle_z[angle_z==360] = 255
        angle_z = (angle_z-angle_z[angle_z<255].min())*(254/(angle_z[angle_z<255].max()-angle_z[angle_z<255].min()))
        # combine three channels and save to a png image
        new_img_angles = np.dstack((angle_x, angle_y))
        new_img_angles = np.dstack((new_img_angles, angle_z))
        new_img_angles = new_img_angles.astype(np.uint8)
    else:
        new_img_angles = np.dstack((angle_x, angle_y))
        new_img_angles = np.dstack((new_img_angles, angle_z))
    
    # if args.vis_img:
    #     signed_x[signed_x==360] = 255
    #     signed_x = (signed_x-signed_x[signed_x<255].min())*(254/(signed_x[signed_x<255].max()-signed_x[signed_x<255].min()))
    #     signed_y[signed_y==360] = 255
    #     signed_y = (signed_y-signed_y[signed_y<255].min())*(254/(signed_y[signed_y<255].max()-signed_y[signed_y<255].min()))
    #     signed_z[signed_z==360] = 255
    #     signed_z = (signed_z-signed_z[signed_z<255].min())*(254/(signed_z[signed_z<255].max()-signed_z[signed_z<255].min()))
    #     # combine three channels and save to a png image
    #     new_img_signed = np.dstack((signed_x, signed_y))
    #     new_img_signed = np.dstack((new_img_signed, signed_z))
    #     new_img_signed = new_img_signed.astype(np.uint8)

    # else:
    #     new_img_signed = np.dstack((signed_x, signed_y))
    #     new_img_signed = np.dstack((new_img_signed, signed_z))
    
    return new_img_angles



# testlist = np.loadtxt(os.path.join('/workspace/DATA/Occ_LineMod','test',args.test_list),dtype='str')

K=np.array([[572.4114, 0.,         325.2611],
            [0.,        573.57043,  242.04899],
            [0.,        0.,         1.]])
# meta_file = open(os.path.join('/workspace/DATA/Occ_Linemod/data',args.cls_num, 'gt.yml'), "r")

# meta_lst = yaml.safe_load(meta_file)

################################################### For pbr data ###########################################
for scene_id in tqdm.tqdm(range(10)):
    scene_id = str(scene_id).zfill(6)
    
    for item_name in tqdm.tqdm(range(1000)):
        item_name = str(item_name).zfill(6)
        
        with Image.open(os.path.join('/workspace/DATA/Occ_LineMod','train_pbr',scene_id,"depth/{}.png".format(item_name))) as di:
            dpt_mm = np.array(di)
        
        
        dpt_mm = dpt_mm * 0.1
        cam_scale = 1000.0
        dpt_mm = fill_missing(dpt_mm, cam_scale, 1)

        dpt_mm = dpt_mm.copy().astype(np.uint16)
        nrm_map = normalSpeed.depth_normal(dpt_mm, K[0][0], K[1][1], 5, 2000, 20, False)

        # if True:
        #     show_nrm_map = ((nrm_map + 1.0) * 127).astype(np.uint8)
        #     img_file = os.path.join('/workspace/DATA/Linemod_preprocessed/data',args.cls_num,'vis_nrm_{}.png'.format(item_name))
        #     cv2.imwrite(img_file, show_nrm_map)
        # dpt_m = dpt_mm.astype(np.float32) / cam_scale
        # dpt_xyz = dpt_2_pcld(dpt_m, 1.0, K)
        # dpt_xyz[np.isnan(dpt_xyz)] = 0.0
        # dpt_xyz[np.isinf(dpt_xyz)] = 0.0

        
        # rgb = pseudo_gen(args, dpt_xyz)
        rgb_nrm = pseudo_nrm_angle( nrm_map)
        
        if args.vis_img:
            
            # img_file = os.path.join('/workspace/DATA/Occ_Linemod/data',args.cls_num,'vis_cur_{}.png'.format(item_name))
            # cv2.imwrite(img_file, surface_curvature)
            # # img_file = os.path.join('/workspace/DATA/Linemod_preprocessed/data',args.cls_num,'vis_signed_{}.png'.format(item_name))
            # # cv2.imwrite(img_file, rgb_s)
            # print('Please look at /workspace/DATA/Occ_Linemod/data/your_class/angles_or_signed_itemname.png')
            # img_file = os.path.join('/workspace/DATA/Occ_Linemod/data',args.cls_num,'vis_angles_{}.png'.format(item_name))
            # cv2.imwrite(img_file, rgb)
            # # img_file = os.path.join('/workspace/DATA/Linemod_preprocessed/data',args.cls_num,'vis_signed_{}.png'.format(item_name))
            # # cv2.imwrite(img_file, rgb_s)
            # print('Please look at /workspace/DATA/Occ_Linemod/data/your_class/angles_or_signed_itemname.png')
            img_file = os.path.join('/workspace/DATA/Occ_LineMod','train_pbr','vis_nrm_angles_{}.png'.format(item_name))
            cv2.imwrite(img_file, rgb_nrm)
            # img_file = os.path.join('/workspace/DATA/Linemod_preprocessed/data',args.cls_num,'vis_signed_{}.png'.format(item_name))
            # cv2.imwrite(img_file, rgb_s)
            print('Please look at /workspace/DATA/Occ_Linemod/data/your_class/angles_or_signed_itemname.png')
            exit()
            
            
        if not os.path.exists(os.path.join('/workspace/DATA/Occ_LineMod','train_pbr',scene_id,'pseudo_nrm_angles')):
            os.makedirs(os.path.join('/workspace/DATA/Occ_LineMod','train_pbr',scene_id,'pseudo_nrm_angles'))
        rgb_file = os.path.join('/workspace/DATA/Occ_LineMod','train_pbr',scene_id,'pseudo_nrm_angles','{}'.format(item_name))
        np.savez_compressed(rgb_file, angles=rgb_nrm)
    print('Finish real/ testing data generation!!')


# K=np.array([[572.4114, 0.0, 325.2611082792282], 
#             [0.0, 573.57043, 242.04899594187737], 
#             [0.0, 0.0, 1.0]])
# depth_scale=0.1
# gt_info_file = "/workspace/DATA/Linemod_preprocessed/000000/scene_gt.json"
# gt_info_f = open(gt_info_file)
# gt_info = json.load(gt_info_f)
# obj_id=2
# for num in gt_info.keys():
#     info = gt_info[num]
#     for idx in range(len(info)):
#         gt_idx = info[idx]
#         if gt_idx['obj_id']==obj_id:
#             print('/workspace/DATA/Linemod_preprocessed/000000/mask/{:06d}_'.format(int(num))+'{:06d}.png'.format(idx))
        
        

# depth_img = "/workspace/DATA/Linemod_preprocessed/000000/depth/000000.png"
# with Image.open(depth_img) as di:
#     dpt_mm = np.array(di)

# dpt_mm = dpt_mm*depth_scale
# dpt_mm = dpt_mm.astype(np.uint16)
# nrm_map = normalSpeed.depth_normal(dpt_mm, K[0][0], K[1][1], 5, 2500, 20, False)
# nrm_angle_vis = pseudo_nrm_angle(nrm_map)
# if True:
#     img_file = "/workspace/DATA/Linemod_preprocessed/000000/nrm_angle_000000.png"
#     cv2.imwrite(img_file, nrm_angle_vis)
#     show_nrm_map = ((nrm_map + 1.0) * 127).astype(np.uint8)
#     img_file = "/workspace/DATA/Linemod_preprocessed/000000/nrm_000000.png"
#     cv2.imwrite(img_file, show_nrm_map)

# Look at histogram

# rgb_list_x = []
# rgbs_list_x = []
# rgb_list_y = []
# rgbs_list_y = []
# rgb_list_z = []
# rgbs_list_z = [] 

# within for loop, put the below codes, to fill rgb_list_xxx and rgbs_list_xxx 
# rgb_list_x.append(rgb[:,:,0].reshape(1, 480*640))
# rgbs_list_x.append(rgb_s[:,:,0].reshape(1, 480*640))
# rgb_list_y.append(rgb[:,:,1].reshape(1, 480*640))
# rgbs_list_y.append(rgb_s[:,:,1].reshape(1, 480*640))
# rgb_list_z.append(rgb[:,:,2].reshape(1, 480*640))
# rgbs_list_z.append(rgb_s[:,:,2].reshape(1, 480*640))    

# then plot the histogram along with each axis.
# plt.hist(rgb_list_x,bins=36,alpha=0.5, label='x')
# plt.hist(rgb_list_y,bins=36,alpha=0.5, label='y')
# plt.hist(rgb_list_z,bins=36,alpha=0.5, label='z')
# plt.legend(loc='upper right')
# plt.savefig(os.path.join('/workspace/DATA/Linemod_preprocessed/data','15', "training_hist_angles.png"))
# plt.cla()
# plt.hist(rgbs_list_x,bins=36,alpha=0.5, label='x_s')
# plt.hist(rgbs_list_y,bins=36,alpha=0.5, label='y_s')
# plt.hist(rgbs_list_z,bins=36,alpha=0.5, label='z_s')
# plt.legend(loc='upper right')
# plt.savefig(os.path.join('/workspace/DATA/Linemod_preprocessed/data','15', "training_hist_signed.png"))

