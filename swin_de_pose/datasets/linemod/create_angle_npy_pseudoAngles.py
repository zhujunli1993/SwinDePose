import numpy as np
import math
from PIL import Image
import yaml
import os
import cv2
import tqdm
from argparse import ArgumentParser
import matplotlib.pyplot as plt
parser = ArgumentParser()
parser.add_argument(
    "--cls_num", type=str, default="06",
    help="Target object from {ape, benchvise, cam, can, cat, driller, duck, \
    eggbox, glue, holepuncher, iron, lamp, phone} (default phone)"
)
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

def pseudo_gen(args, dpt_xyz):
    height=480
    width=640
    # Set up-axis 
    x_up = np.array([1.0, 0.0, 0.0])
    y_up = np.array([0.0, 1.0, 0.0])
    z_up = np.array([0.0, 0.0, 1.0])

    angle_x = []
    signed_x = []
    angle_y = []
    signed_y = []
    angle_z = []
    signed_z = []
    dpt_xyz = np.reshape(dpt_xyz,(height, width, 3))
    for i in range(0, height):
        for j in range(1, width):
            p_2 = dpt_xyz[i, j]
            p_1 = dpt_xyz[i, j-1]
            if p_2[0]+p_2[1]+p_2[2]==0.0 or p_1[0]+p_1[1]+p_1[2]==0.0:
                angle_x.append(360.0)
                angle_y.append(360.0)
                angle_z.append(360.0)
                signed_x.append(360.0)
                signed_y.append(360.0)
                signed_z.append(360.0)
                continue
            else:
                difference = p_2 - p_1
                difference_lengh = np.sqrt(math.pow(difference[0],2)+math.pow(difference[1],2)+math.pow(difference[2],2))
                epsilon = 1E-6
                if difference_lengh < epsilon:
                    angle_x.append(360.0)
                    angle_y.append(360.0)
                    angle_z.append(360.0)
                    signed_x.append(360.0)
                    signed_y.append(360.0)
                    signed_z.append(360.0)
                    continue
                else:
                    value_x = (difference[0]*x_up[0] + difference[1]*x_up[1] + difference[2]*x_up[2]) / difference_lengh
                    value_y = (difference[0]*y_up[0] + difference[1]*y_up[1] + difference[2]*y_up[2]) / difference_lengh
                    value_z = (difference[0]*z_up[0] + difference[1]*z_up[1] + difference[2]*z_up[2]) / difference_lengh
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
                    
                    if j == 1:
                        signed_x.append(360.0)
                        signed_y.append(360.0)
                        signed_z.append(360.0)
                        continue
                    else:
                        
                        p_0 = dpt_xyz[i, j-2]
                    
                        if p_0[0]+p_0[1]+p_0[2]==0.0:
                            signed_x.append(360.0)
                            signed_y.append(360.0)
                            signed_z.append(360.0)
                            continue
                        else:
                            dot_prod = difference[0]*(p_1[0]-p_0[0]) + difference[1]*(p_1[1]-p_0[1]) + difference[2]*(p_1[2]-p_0[2])
                            if dot_prod >= 0.0:
                                signed_x.append(math.acos(value_x)*180.0/math.pi)
                            else:
                                signed_x.append(-1*math.acos(value_x)*180.0/math.pi)
                            if dot_prod >= 0.0:
                                signed_y.append(math.acos(value_y)*180.0/math.pi)
                            else:
                                signed_y.append(-1*math.acos(value_y)*180.0/math.pi)
                            if dot_prod >= 0.0:
                                signed_z.append(math.acos(value_z)*180.0/math.pi)
                            else:
                                signed_z.append(-1*math.acos(value_z)*180.0/math.pi)
        angle_x.append(360.0)
        angle_y.append(360.0)
        angle_z.append(360.0)
        signed_x.append(360.0)
        signed_y.append(360.0)
        signed_z.append(360.0)
        
    angle_x = np.reshape(angle_x, [height, width])
    signed_x = np.reshape(signed_x, [height, width])
    angle_y = np.reshape(angle_y, [height, width])
    signed_y = np.reshape(signed_y, [height, width])
    angle_z = np.reshape(angle_z, [height, width])
    signed_z = np.reshape(signed_z, [height, width])

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
    
    if args.vis_img:
        signed_x[signed_x==360] = 255
        signed_x = (signed_x-signed_x[signed_x<255].min())*(254/(signed_x[signed_x<255].max()-signed_x[signed_x<255].min()))
        signed_y[signed_y==360] = 255
        signed_y = (signed_y-signed_y[signed_y<255].min())*(254/(signed_y[signed_y<255].max()-signed_y[signed_y<255].min()))
        signed_z[signed_z==360] = 255
        signed_z = (signed_z-signed_z[signed_z<255].min())*(254/(signed_z[signed_z<255].max()-signed_z[signed_z<255].min()))
        # combine three channels and save to a png image
        new_img_signed = np.dstack((signed_x, signed_y))
        new_img_signed = np.dstack((new_img_signed, signed_z))
        new_img_signed = new_img_signed.astype(np.uint8)

    else:
        new_img_signed = np.dstack((signed_x, signed_y))
        new_img_signed = np.dstack((new_img_signed, signed_z))
    
    return new_img_angles, new_img_signed


trainlist = np.loadtxt(os.path.join('/workspace/DATA/Linemod_preprocessed/data',args.cls_num,args.train_list),dtype='str')
testlist = np.loadtxt(os.path.join('/workspace/DATA/Linemod_preprocessed/data',args.cls_num,args.test_list),dtype='str')

K=np.array([[572.4114, 0.,         325.2611],
            [0.,        573.57043,  242.04899],
            [0.,        0.,         1.]])
meta_file = open(os.path.join('/workspace/DATA/Linemod_preprocessed/data',args.cls_num, 'gt.yml'), "r")
meta_lst = yaml.safe_load(meta_file)

for item_name in tqdm.tqdm(trainlist):
    with Image.open(os.path.join('/workspace/DATA/Linemod_preprocessed/data',args.cls_num, "depth/{}.png".format(item_name))) as di:
        dpt_mm = np.array(di)

    meta = meta_lst[int(item_name)]
    meta = meta[0]
    
    cam_scale = 1000.0

    dpt_mm = dpt_mm.copy().astype(np.uint16)
    dpt_m = dpt_mm.astype(np.float32) / cam_scale
    dpt_xyz = dpt_2_pcld(dpt_m, 1.0, K)
    dpt_xyz[np.isnan(dpt_xyz)] = 0.0
    dpt_xyz[np.isinf(dpt_xyz)] = 0.0
    rgb, rgb_s = pseudo_gen(args, dpt_xyz)
    if args.vis_img:
        img_file = os.path.join('/workspace/DATA/Linemod_preprocessed/data',args.cls_num,'vis_angles_{}.png'.format(item_name))
        cv2.imwrite(img_file, rgb)
        img_file = os.path.join('/workspace/DATA/Linemod_preprocessed/data',args.cls_num,'vis_signed_{}.png'.format(item_name))
        cv2.imwrite(img_file, rgb_s)
        print('Please look at /workspace/DATA/Linemod_preprocessed/data/your_class/angles_or_signed_itemname.png')
        exit()
    rgb_file = os.path.join('/workspace/DATA/Linemod_preprocessed/data',args.cls_num,'pseudo_angles_signed','{}'.format(item_name) )
    np.savez_compressed(rgb_file, angles=rgb, signed=rgb_s)
print('Finish real/ training data generation!!')



for item_name in tqdm.tqdm(testlist):
    with Image.open(os.path.join('/workspace/DATA/Linemod_preprocessed/data',args.cls_num, "depth/{}.png".format(item_name))) as di:
        dpt_mm = np.array(di)

    meta = meta_lst[int(item_name)]
    meta = meta[0]
    
    cam_scale = 1000.0
    dpt_mm = dpt_mm.copy().astype(np.uint16)
    dpt_m = dpt_mm.astype(np.float32) / cam_scale
    dpt_xyz = dpt_2_pcld(dpt_m, 1.0, K)
    dpt_xyz[np.isnan(dpt_xyz)] = 0.0
    dpt_xyz[np.isinf(dpt_xyz)] = 0.0
    rgb, rgb_s = pseudo_gen(args, dpt_xyz)
    if args.vis_img:
        img_file = os.path.join('/workspace/DATA/Linemod_preprocessed/data',args.cls_num,'vis_angles_{}.png'.format(item_name))
        cv2.imwrite(img_file, rgb)
        img_file = os.path.join('/workspace/DATA/Linemod_preprocessed/data',args.cls_num,'vis_signed_{}.png'.format(item_name))
        cv2.imwrite(img_file, rgb_s)
        print('Please look at /workspace/DATA/Linemod_preprocessed/data/your_class/angles_or_signed_itemname.png')
        exit()
    rgb_file = os.path.join('/workspace/DATA/Linemod_preprocessed/data',args.cls_num,'pseudo_angles_signed','{}'.format(item_name) )
    np.savez_compressed(rgb_file, angles=rgb, signed=rgb_s)
print('Finish real/ testing data generation!!')


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

