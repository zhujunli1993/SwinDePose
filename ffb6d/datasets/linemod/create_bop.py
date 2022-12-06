import numpy as np
import json
import argparse
import os
# K=np.array([[572.4114, 0.0, 325.2611082792282], 
#             [0.0, 573.57043, 242.04899594187737], 
#             [0.0, 0.0, 1.0]])
# depth_scale=0.1
parser = argparse.ArgumentParser()
parser.add_argument('--cls', type=int, default="15")
parser.add_argument('--range', type=int,default=28)


args = parser.parse_args()

file_list = []
plst_pth = os.path.join("/workspace/DATA/Linemod_preprocessed/train_pbr", "train_pbr_obj_"+str(args.cls)+".txt")
for i_ran in range(args.range):

    gt_info_file = "/workspace/DATA/Linemod_preprocessed/train_pbr/{:06d}".format(i_ran)+"/scene_gt.json"
    gt_info_f = open(gt_info_file)
    gt_info = json.load(gt_info_f)
    obj_id=args.cls
    


    for num in gt_info.keys():
        info = gt_info[num]
        for idx in range(len(info)):
            gt_idx = info[idx]
            if gt_idx['obj_id']==obj_id:

                file_list.append('/workspace/DATA/Linemod_preprocessed/train_pbr/{:06d}'.format(i_ran)+'/mask/{:06d}_'.format(int(num))+'{:06d}.png'.format(idx))
            
        
with open(plst_pth, 'w') as of:
    for pth in file_list:
        print(pth, file=of)
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