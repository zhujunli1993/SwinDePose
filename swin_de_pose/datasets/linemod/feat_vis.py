import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter


def scale_pseudo(pseudo):
    pseudo = (pseudo-pseudo.min())*(255/(pseudo.max()-pseudo.min()))
    return pseudo

test_lst = np.loadtxt('/workspace/DATA/Linemod_preprocessed/data/15/test.txt')
test_file_ind = str(int(test_lst[0])).zfill(4)
rgb_img_pth = os.path.join('/workspace/DATA/Linemod_preprocessed/data','15','rgb',test_file_ind+'.png')

with Image.open(rgb_img_pth) as ri:
    rgb_img = np.array(ri)
    

# Load point clouds
pc = torch.load('/workspace/REPO/pose_estimation/ffb6d/train_log/lm_swinTiny_phone_fullSyn_dense_fullInc/phone/255_pc.pt')
pc = pc.cpu().detach().numpy()
pc=pc.squeeze()
pc = np.transpose(pc)
pc_coor = pc[:,0:3]
np.savetxt('/workspace/REPO/pose_estimation/ffb6d/train_log/lm_swinTiny_phone_fullSyn_dense_fullInc/phone/255_pc.txt',pc, delimiter=',')

pc_feat = torch.load('/workspace/REPO/pose_estimation/ffb6d/train_log/lm_swinTiny_phone_fullSyn_dense_fullInc/phone/255_img_pc.pt')
pc_feat = pc_feat.cpu().detach().numpy()
pc_feat=pc_feat.squeeze()
pc_feat = np.transpose(pc_feat)
pc_feat = np.absolute(pc_feat)
pc_feat = np.mean(pc_feat,axis=1)
np.savetxt('/workspace/REPO/pose_estimation/ffb6d/train_log/lm_swinTiny_phone_fullSyn_dense_fullInc/phone/255_img_pc_mean.txt',pc_feat, delimiter=',')

pc_feat=np.expand_dims(pc_feat,axis=1)
pc_img_feat=np.hstack((pc_coor,pc_feat))
np.savetxt('/workspace/REPO/pose_estimation/ffb6d/train_log/lm_swinTiny_phone_fullSyn_dense_fullInc/phone/255_img_pc_feat.txt',pc_img_feat, delimiter=',')

import pdb;pdb.set_trace()
img_final=torch.load('/workspace/REPO/pose_estimation/ffb6d/train_log/lm_swinTiny_phone_fullSyn_dense_fullInc/phone/255_img_final.pt')
pc_final=torch.load('/workspace/REPO/pose_estimation/ffb6d/train_log/lm_swinTiny_phone_fullSyn_dense_fullInc/phone/255_pc_final.pt')
img_final=img_final.cpu().detach().numpy()
img_final=img_final.squeeze()
pc_final=pc_final.cpu().detach().numpy()
pc_final=pc_final.squeeze()
img_final=np.transpose(img_final)
pc_final=np.transpose(pc_final)
img_final=np.absolute(img_final)
pc_final=np.absolute(pc_final)
img_final=np.mean(img_final,axis=1)
pc_final=np.mean(pc_final,axis=1)
img_final=np.expand_dims(img_final,axis=1)
pc_final=np.expand_dims(pc_final,axis=1)
pc_imgfinal_feat=np.hstack((pc_coor,img_final))
pc_pcfinal_feat=np.hstack((pc_coor,pc_final))
np.savetxt('/workspace/REPO/pose_estimation/ffb6d/train_log/lm_swinTiny_phone_fullSyn_dense_fullInc/phone/255_pc_imgfinal_feat.txt',pc_imgfinal_feat, delimiter=',')
np.savetxt('/workspace/REPO/pose_estimation/ffb6d/train_log/lm_swinTiny_phone_fullSyn_dense_fullInc/phone/255_pc_pcfinal_feat.txt',pc_pcfinal_feat, delimiter=',')
# gt_kp = np.load('/workspace/REPO/pose_estimation/ffb6d/train_log/lm_swinTiny_phone_fullSyn_dense_fullInc/phone/eval_results/gt_kp.npy')
# gt_kp=gt_kp[0]
# np.savetxt('/workspace/REPO/pose_estimation/ffb6d/train_log/lm_swinTiny_phone_fullSyn_dense_fullInc/phone/gt_kp.txt',gt_kp,delimiter=',')
# pred_kp = np.load('/workspace/REPO/pose_estimation/ffb6d/train_log/lm_swinTiny_phone_fullSyn_dense_fullInc/phone/eval_results/pred_kp.npy')
# pred_kp=pred_kp[0]
# np.savetxt('/workspace/REPO/pose_estimation/ffb6d/train_log/lm_swinTiny_phone_fullSyn_dense_fullInc/phone/pred_kp.txt',pred_kp,delimiter=',')
# gt_ctr = np.load('/workspace/REPO/pose_estimation/ffb6d/train_log/lm_swinTiny_phone_fullSyn_dense_fullInc/phone/eval_results/gt_ctr.npy')
# gt_ctr=gt_ctr[0]
# np.savetxt('/workspace/REPO/pose_estimation/ffb6d/train_log/lm_swinTiny_phone_fullSyn_dense_fullInc/phone/gt_ctr.txt',gt_ctr,delimiter=',')
# cld = np.load('/workspace/REPO/pose_estimation/ffb6d/train_log/lm_swinTiny_phone_fullSyn_dense_fullInc/phone/eval_results/cld.npy')
# feat = torch.load('/workspace/REPO/pose_estimation/ffb6d/train_log/lm_swinTiny_phone_fullSyn_dense_fullInc/phone/0_img.pt')
# feat = feat.cpu().detach().numpy()
# feat = feat.squeeze()
# feat = np.sum(feat, axis=0)
# feat = feat / np.max(feat)  # Normalization
# mask = Image.fromarray(feat * 255)
# # mask = mask.resize(size=(480,640), resample=Image.LANCZOS)
# mask = mask.convert("L")
# mask = mask.filter(ImageFilter.GaussianBlur(radius=16))
# mask = np.array(mask).astype(float)
# mask -= np.min(mask)
# mask /= np.max(mask)
# #plt.save(np.zeros_like(rgb_img), vmin=0, vmax=0.5)
# # img = np.concatenate([rgb_img, mask[..., np.newaxis]], axis=2)
# # img = Image.fromarray(img)
# # img.save('/workspace/REPO/pose_estimation/ffb6d/train_log/lm_swinTiny_phone_fullSyn_dense_fullInc/phone/feat_img_1.png')
# plt.imshow(np.zeros_like(rgb_img), vmin=0, vmax=0.5)
# img = np.concatenate([rgb_img, mask[..., np.newaxis]], axis=2)
# plt.savefig('foo.png',img.astype(np.uint8))


# avg_pool_feat = torch.sum(feat, dim=1).squeeze()
# avg_pool_feat = avg_pool_feat.cpu().detach().numpy()
# avg_pool_feat = scale_pseudo(avg_pool_feat)
# avg_pool_feat = np.uint8(avg_pool_feat)
# feat_msk = np.zeros((480,640,3),dtype=np.uint8)
# for i in range(3):
#     q[:,:,i] = avg_pool_feat


# feat_msk = Image.fromarray(feat_msk)
# feat_msk.save('/workspace/REPO/pose_estimation/ffb6d/train_log/lm_swinTiny_phone_fullSyn_dense_fullInc/phone/feat_img_0.png')

