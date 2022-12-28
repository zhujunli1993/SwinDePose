import numpy as np
import yaml
import pickle as pkl
import json
import os
import shutil
scene_gt_pth = '/workspace/DATA/Occ_Linemod/test/000002/scene_gt.json'
scene_gt_info_path = '/workspace/DATA/Occ_Linemod/test/000002/scene_gt_info.json'
scene_camera_path = '/workspace/DATA/Occ_Linemod/test/000002/scene_camera.json'
f = open(scene_gt_pth)
data = json.load(f)
f_bb = open(scene_gt_info_path)
obj_bbs = json.load(f_bb)
f_cam = open(scene_camera_path)
cam_info = json.load(f_cam)
dict_file={}
for ind in range(len(data)):

    index = str(ind)
    content = data[index]
    bb_contents = obj_bbs[index]
    dict_file[index]=[]
    for o_idx in range(len(content)):

        cam_R_m2c=content[o_idx]["cam_R_m2c"]
        cam_t_m2c=content[o_idx]["cam_t_m2c"]
        obj_id=content[o_idx]["obj_id"]
        obj_bb=bb_contents[o_idx]["bbox_obj"]
        group = {'cam_R_m2c':cam_R_m2c,'cam_t_m2c':cam_t_m2c,'obj_bb':obj_bb,'obj_id':obj_id}
        dict_file[index].append(group)
        
with open('gt.yml', 'w') as yaml_file:
    yaml.dump(dict_file, yaml_file, default_flow_style=False)
dict_file={}
for ind in range(len(cam_info)):
    index = str(ind)
    content = cam_info[index]
    cam_K=content['cam_K']
    depth_scale=content['depth_scale']
    dict_file[index]=[]
    group = {'cam_K':cam_K,'depth_scale':depth_scale}
    dict_file[index].append(group)
with open('info.yml', 'w') as yaml_file:
    yaml.dump(dict_file, yaml_file, default_flow_style=False)
obj_ids = [1, 5, 6, 8, 9, 10, 11, 12]
for ind in range(len(obj_ids)):
    obj_id = obj_ids[ind]
    obj_fld = "/workspace/DATA/Occ_Linemod/test/{:02d}".format(obj_id)
    if not os.path.exists("/workspace/DATA/Occ_Linemod/test/{:02d}".format(obj_id)):
        os.makedirs(obj_fld)
    dp_fld = os.path.join(obj_fld,'depth')
    if not os.path.exists(os.path.join(obj_fld,'depth')):
        os.makedirs(dp_fld)
    msk_fld = os.path.join(obj_fld,'mask')
    if not os.path.exists(os.path.join(obj_fld,'mask')):
        os.makedirs(msk_fld)
    rgb_fld = os.path.join(obj_fld,'rgb')
    if not os.path.exists(os.path.join(obj_fld,'rgb')):
        os.makedirs(rgb_fld)
    gt_pth = 'gt.yml'
    if not os.path.exists(obj_fld+"/gt.yml"):
        shutil.copyfile(gt_pth, obj_fld+"/gt.yml")
    info_pth = 'info.yml'
    if not os.path.exists(obj_fld+"/info.yml"):
        shutil.copyfile(info_pth, obj_fld+"/info.yml")
    
    files = os.listdir(obj_fld+"/mask")
    ids = []

    for scene_id in files:
        id = scene_id.split('.')[0]
        ids.append(str(id))
        
    np.savetxt(obj_fld+"/test.txt", ids, fmt="%s")
    for ind_ in range(len(data)):
        msk_file_pth = "/workspace/DATA/Occ_Linemod/test/000002"+"/mask/{:06d}_".format(ind_)+"{:06d}.png".format(ind)
        msk_dst_pth = obj_fld+'/mask/{:04d}.png'.format(ind_)
        if os.path.exists(msk_file_pth) and not os.path.exists(msk_dst_pth):
            shutil.copyfile(msk_file_pth, msk_dst_pth)
        rgb_file_pth = "/workspace/DATA/Occ_Linemod/test/000002"+"/rgb/{:06d}.png".format(ind_)
        rgb_dst_pth = obj_fld+'/rgb/{:04d}.png'.format(ind_)
        if os.path.exists(rgb_file_pth) and not os.path.exists(rgb_dst_pth):
            shutil.copyfile(rgb_file_pth, rgb_dst_pth)
        
        dp_file_pth = "/workspace/DATA/Occ_Linemod/test/000002"+"/depth/{:06d}.png".format(ind_)
        dp_dst_pth = obj_fld+'/depth/{:04d}.png'.format(ind_)
        if os.path.exists(dp_file_pth) and not os.path.exists(dp_dst_pth):
            shutil.copyfile(dp_file_pth, dp_dst_pth)
        