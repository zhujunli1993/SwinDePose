
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
import scipy.io as scio
import scipy.misc
import depth_map_utils_ycb as depth_map_utils
parser = ArgumentParser()


parser.add_argument(
    '--train_list', type=str, 
    help="training list for real/ generation."
)
parser.add_argument(
    '--test_list', type=str, 
    help="testing list for real/ generation."
)
parser.add_argument(
    '--vis_img', action="store_true",
    help="visulaize histogram."
)
args = parser.parse_args()
trainlist = np.loadtxt(os.path.join('/workspace/DATA/YCBV',args.train_list),dtype='str')
testlist = np.loadtxt(os.path.join('/workspace/DATA/YCBV',args.test_list),dtype='str')
for item_name in tqdm.tqdm(trainlist):
    
    f_idx = item_name.split('/')[1]
    
    data_type = item_name.split('/')[0]
    if data_type != 'data_syn':
        continue
        i_idx = item_name.split('/')[2]
        meta = scio.loadmat(os.path.join('/workspace/DATA/YCBV',data_type, f_idx, i_idx+'-meta.mat'))
        cam_scale = meta['factor_depth'].astype(np.float32)[0][0]
        if not cam_scale==10000.:
            print(f_idx)
            print(i_idx)
    else:
        if not os.path.exists(os.path.join('/workspace/DATA/YCBV',data_type, f_idx+'-meta.mat')):
            print(os.path.join('/workspace/DATA/YCBV',data_type, f_idx+'-meta.mat'))
        if not os.path.exists(os.path.join('/workspace/DATA/YCBV',data_type, f_idx+'-depth.png')):
            print(os.path.join('/workspace/DATA/YCBV',data_type, f_idx+'-depth.png'))
        if not os.path.exists(os.path.join('/workspace/DATA/YCBV',data_type, f_idx+'-color.png')):
            print(os.path.join('/workspace/DATA/YCBV',data_type, f_idx+'-color.png'))
        if not os.path.exists(os.path.join('/workspace/DATA/YCBV',data_type, f_idx+'-label.png')):
            print(os.path.join('/workspace/DATA/YCBV',data_type, f_idx+'-label.png'))
        if os.path.exists(os.path.join('/workspace/DATA/YCBV',data_type, f_idx+'-meta.mat')):
            meta = scio.loadmat(os.path.join('/workspace/DATA/YCBV',data_type, f_idx+'-meta.mat'))
            cam_scale = meta['factor_depth'].astype(np.float32)[0][0]
            if not cam_scale==10000.:
                print(f_idx)
