import numpy as np
import cv2 
import matplotlib.pyplot as plt
import os

path = '/workspace/REPO/pose_estimation/ffb6d/train_log/lm_6_pseudo_noSyn_depth_RGB_mlp/phone/eval_results/test_res.npz'
content = np.load(path,allow_pickle=True)
import pdb; pdb.set_trace()
angle=content['angles']
sign=content['signed']
# cv2.imwrite('/workspace/DATA/Linemod_preprocessed/fuse/phone/0_rgb.png',rgb)
# angle = np.reshape(angle, (480*640*3,1))
# angle = [a for a in angle if a != 360.]
angle[:,:,0][angle[:,:,0]==360] = 255
angle[:,:,0][angle[:,:,0]<255] = (angle[:,:,0][angle[:,:,0]<255]-angle[:,:,0][angle[:,:,0]<255].min())*(254/(angle[:,:,0][angle[:,:,0]<255].max()-angle[:,:,0][angle[:,:,0]<255].min()))
angle[:,:,1][angle[:,:,1]==360] = 255
angle[:,:,1][angle[:,:,1]<255] = (angle[:,:,1][angle[:,:,1]<255]-angle[:,:,1][angle[:,:,1]<255].min())*(254/(angle[:,:,1][angle[:,:,1]<255].max()-angle[:,:,1][angle[:,:,1]<255].min()))
angle[:,:,2][angle[:,:,2]==360] = 255
angle[:,:,2][angle[:,:,2]<255] = (angle[:,:,2][angle[:,:,2]<255]-angle[:,:,2][angle[:,:,2]<255].min())*(254/(angle[:,:,2][angle[:,:,2]<255].max()-angle[:,:,2][angle[:,:,2]<255].min()))
cv2.imwrite('/workspace/DATA/Linemod_preprocessed/data/15/pseudo_angles_signed/0.png',angle)

npz_img = cv2.imread('/workspace/DATA/Linemod_preprocessed/data/15/pseudo_angles_signed/0.png')
# plt.hist(angle,bins=36,alpha=0.5, label='0.npz')
# plt.savefig(os.path.join('/workspace/DATA/Linemod_preprocessed/renders/phone', "training_hist_angles.png"))