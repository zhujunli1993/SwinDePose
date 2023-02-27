import numpy as np
import yaml
import pickle as pkl
import json
import os


lmo_obj_dict={
    'ape':1,
    'can':5,
    'cat':6,
    'driller':8,
    'duck':9,
    'eggbox':10,
    'glue':11,
    'holepuncher':12
}
# for obj_name, obj_id_ind in lmo_obj_dict.items():
#     masks_pths = []

#     scene_id=2
#     scene_id = str(scene_id).zfill(6)
#     scene_gt_pth = os.path.join('/workspace/DATA/Occ_LineMod/test',scene_id, 'scene_gt.json')
#     scene_gt_info_path = os.path.join('/workspace/DATA/Occ_LineMod/test',scene_id, 'scene_gt_info.json')
#     # scene_camera_path = os.path.join('/workspace/DATA/Occ_LineMod/train_pbr/plato',scene_id, 'scene_camera.json')
#     f = open(scene_gt_pth)
#     data = json.load(f)
#     f_bb = open(scene_gt_info_path)
#     obj_bbs = json.load(f_bb)
#     # f_cam = open(scene_camera_path)
#     # cam_info = json.load(f_cam)
#     dict_file={}
#     for ind in range(len(data)):
#         index = str(ind)
#         content = data[index]
#         bb_contents = obj_bbs[index]
#         dict_file[index]=[]
#         for o_idx in range(len(content)):
#             # cam_R_m2c=content[o_idx]["cam_R_m2c"]
#             # cam_t_m2c=content[o_idx]["cam_t_m2c"]
#             obj_id=content[o_idx]["obj_id"]
#             if obj_id == obj_id_ind and bb_contents[o_idx]["visib_fract"]!=0.0:
                
#                 file_id = index.zfill(6)
#                 mask_id = str(o_idx).zfill(6)
#                 mask_pth = os.path.join('/workspace/DATA/Occ_LineMod/test',scene_id,'mask_visib',file_id+'_'+mask_id+'.png')
#                 masks_pths.append(mask_pth)


#     np.savetxt('/workspace/DATA/Occ_LineMod/test/testing_msk_'+obj_name+'.txt', masks_pths, fmt='%s')
for obj_name, obj_id_ind in lmo_obj_dict.items():
    masks_pths = []

    for scene_id in range(0,50):
        scene_id = str(scene_id).zfill(6)
        scene_gt_pth = os.path.join('/workspace/DATA/Occ_LineMod/train_pbr',scene_id, 'scene_gt.json')
        scene_gt_info_path = os.path.join('/workspace/DATA/Occ_LineMod/train_pbr',scene_id, 'scene_gt_info.json')
        # scene_camera_path = os.path.join('/workspace/DATA/Occ_LineMod/train_pbr/plato',scene_id, 'scene_camera.json')
        f = open(scene_gt_pth)
        data = json.load(f)
        f_bb = open(scene_gt_info_path)
        obj_bbs = json.load(f_bb)
        # f_cam = open(scene_camera_path)
        # cam_info = json.load(f_cam)
        dict_file={}
        for ind in range(len(data)):
            index = str(ind)
            content = data[index]
            bb_contents = obj_bbs[index]
            dict_file[index]=[]
            for o_idx in range(len(content)):
                # cam_R_m2c=content[o_idx]["cam_R_m2c"]
                # cam_t_m2c=content[o_idx]["cam_t_m2c"]
                obj_id=content[o_idx]["obj_id"]
                if obj_id == obj_id_ind and bb_contents[o_idx]["visib_fract"]!=0.0:
                    
                    file_id = index.zfill(6)
                    mask_id = str(o_idx).zfill(6)
                    mask_pth = os.path.join('/workspace/DATA/Occ_LineMod/train_pbr',scene_id,'mask_visib',file_id+'_'+mask_id+'.png')
                    masks_pths.append(mask_pth)


    np.savetxt('/workspace/DATA/Occ_LineMod/train_pbr/training_msk_'+obj_name+'.txt', masks_pths, fmt='%s')
            