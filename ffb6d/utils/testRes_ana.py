import numpy as np
import cv2 
import matplotlib.pyplot as plt
import os

def get_loss(name):
    path = os.path.join('/workspace/REPO/pose_estimation/ffb6d/train_log',name,'phone/eval_results')
    loss_all_path = os.path.join(path, 'loss_all.txt')
    loss_ctr_path = os.path.join(path, 'loss_ctr.txt')
    loss_kp_path = os.path.join(path, 'loss_kp.txt')
    loss_seg_path = os.path.join(path, 'loss_seg.txt')
    add_path = os.path.join(path, 'add.txt')
    adds_path = os.path.join(path, 'adds.txt')
    gt_ctr_path = os.path.join(path, 'gt_ctr.npy')
    gt_kp_path = os.path.join(path, 'gt_kp.npy')
    pred_kp_path = os.path.join(path, 'pred_kp.npy')
    # imgids_path = os.path.join(path, 'img_ids.txt')

    loss_all = np.loadtxt(loss_all_path)
    loss_ctr = np.loadtxt(loss_ctr_path)
    loss_kp = np.loadtxt(loss_kp_path)
    loss_seg = np.loadtxt(loss_seg_path)
    
    add = np.loadtxt(add_path)
    adds = np.loadtxt(adds_path)
    pred_kp = np.load(pred_kp_path)
    gt_kp = np.load(gt_kp_path)
    gt_ctr = np.load(gt_ctr_path)
    return loss_all, loss_ctr, loss_kp, loss_seg, add, adds, pred_kp, gt_kp, gt_ctr, path

# import pdb; pdb.set_trace()
# cld = np.load('/workspace/REPO/pose_estimation/ffb6d/train_log/lm_6_pseudo_noSyn_depth_RGB_mlp/phone/eval_results/cld.npy')    

   
loss_all, loss_ctr, loss_kp, loss_seg, add, adds, pred_kp, gt_kp, gt_ctr, path = get_loss('lm_5_pseudo_noSyn_addDepth_attention')
loss_all_1, loss_ctr_1, loss_kp_1, loss_seg_1, add_1, adds_1,pred_kp_1, gt_kp_1, gt_ctr_1, path_1 = get_loss('lm_6_pseudo_noSyn_depth_RGB_mlp')


img_ids = np.arange(0,len(loss_all))

# ### Draw loss histograms
# fig, ax = plt.subplots(2,2)
# n_bins = 15

# ax[0,0].hist(loss_all, n_bins, alpha=0.5, density=True, label='lm_5')    
# ax[0,0].hist(loss_all_1, n_bins, alpha=0.5, density=True, label='lm_6') 
# ax[0,0].title.set_text('loss_all')
# ax[0,0].legend(loc='upper right')

# ax[0,1].hist(loss_ctr, n_bins, alpha=0.5, density=True, label='lm_5')    
# ax[0,1].hist(loss_ctr_1, n_bins, alpha=0.5, density=True, label='lm_6')
# ax[0,1].title.set_text('loss_ctr')
# ax[0,1].legend(loc='upper right')

# ax[1,0].hist(loss_kp, n_bins, alpha=0.5, density=True, label='lm_5')  
# ax[1,0].hist(loss_kp_1, n_bins, alpha=0.5, density=True, label='lm_6')  
# ax[1,0].title.set_text('loss_kp')
# ax[1,0].legend(loc='upper right')

# ax[1,1].hist(loss_seg, n_bins, alpha=0.5, density=True, label='lm_5')    
# ax[1,1].hist(loss_seg_1, n_bins, alpha=0.5, density=True, label='lm_6')  
# ax[1,1].title.set_text('loss_seg')
# ax[1,1].legend(loc='upper right')

# fig.tight_layout()
# loss_all_fig_hist = os.path.join(path, 'loss_all_hist_onlyMask.png')
# plt.savefig(loss_all_fig_hist)
# plt.clf()


# ### Draw add/adds histograms

# fig, ax = plt.subplots(1,2)
# n_bins = 15

# ax[0].hist(add, n_bins, alpha=0.5, density=True, label='lm_5')    
# ax[0].hist(add_1, n_bins, alpha=0.5, density=True, label='lm_6') 
# ax[0].title.set_text('add')
# ax[0].legend(loc='right')

# ax[1].hist(adds, n_bins, alpha=0.5, density=True, label='lm_5')    
# ax[1].hist(adds_1, n_bins, alpha=0.5, density=True, label='lm_6')
# ax[1].title.set_text('adds')
# ax[1].legend(loc='right')

# fig.tight_layout()
# add_adds_fig_hist = os.path.join(path, 'add_adds_hist_onlyMask.png')
# plt.savefig(add_adds_fig_hist)
# plt.clf()

# ### Draw loss bar plots    
# fig, ax = plt.subplots(2,2)

# ax[0,0].bar(img_ids, loss_all)
# ax[0,0].title.set_text('loss_all')

# ax[0,1].bar(img_ids, loss_ctr)
# ax[0,1].title.set_text('loss_ctr')

# ax[1,0].bar(img_ids, loss_kp)
# ax[1,0].title.set_text('loss_kp')

# ax[1,1].bar(img_ids, loss_seg)
# ax[1,1].title.set_text('loss_seg')

# fig.tight_layout()
# loss_all_fig = os.path.join(path, 'loss_all_onlyMask.png')
# plt.savefig(loss_all_fig)
# plt.clf()


# ### Draw add/adds bar plots
# fig, ax = plt.subplots(1,2)

# ax[0].bar(img_ids, add)
# ax[0].title.set_text('add')

# ax[1].bar(img_ids, adds)
# ax[1].title.set_text('adds')

# fig.tight_layout()
# add_fig = os.path.join(path, 'add_adds_onlyMask.png')
# plt.savefig(add_fig)
# plt.clf()

# ### Draw loss VS add/adds scatter correlation
# fig, ax = plt.subplots(2,2)
# ax[0][0].scatter(loss_all, add, alpha=0.2, label='lm_5')
# ax[0][0].scatter(loss_all_1, add_1, alpha=0.2, label='lm_6')
# ax[0][0].legend(loc='upper left')
# ax[0][0].title.set_text('loss_all - add')
# ax[0][0].set_xlabel('loss_all')
# ax[0][0].set_ylabel('add')

# ax[0][1].scatter(loss_ctr, add, alpha=0.2, label='lm_5')
# ax[0][1].scatter(loss_ctr_1, add_1, alpha=0.2, label='lm_6')
# ax[0][1].legend(loc='upper left')
# ax[0][1].title.set_text('loss_ctr - add')
# ax[0][1].set_xlabel('loss_ctr')
# ax[0][1].set_ylabel('add')

# ax[1][0].scatter(loss_kp, add, alpha=0.2, label='lm_5')
# ax[1][0].scatter(loss_kp_1, add_1, alpha=0.2, label='lm_6')
# ax[1][0].legend(loc='upper left')
# ax[1][0].title.set_text('loss_kp - add')
# ax[1][0].set_xlabel('loss_kp')
# ax[1][0].set_ylabel('add')

# ax[1][1].scatter(loss_seg, add, alpha=0.2, label='lm_5')
# ax[1][1].scatter(loss_seg_1, add_1, alpha=0.2, label='lm_6')
# ax[1][1].legend(loc='upper left')
# ax[1][1].title.set_text('loss_seg - add')
# ax[1][1].set_xlabel('loss_seg')
# ax[1][1].set_ylabel('add')

# fig.tight_layout()
# sca_fig = os.path.join(path, 'scatter_add_onlyMask.png')
# plt.savefig(sca_fig)
# plt.clf()

### Process predicted kp vs gt kp and gt ctr.
gt_ctr = np.expand_dims(gt_ctr, axis=1)
kp_ctr = np.concatenate((gt_kp, gt_ctr), axis=1)
gt_ctr_1 = np.expand_dims(gt_ctr_1, axis=1)
kp_ctr_1 = np.concatenate((gt_kp_1, gt_ctr_1), axis=1)

dist = pred_kp - kp_ctr
dist_1 = pred_kp_1 - kp_ctr_1

norm = np.linalg.norm(dist, axis=-1)
norm_1 = np.linalg.norm(dist_1, axis=-1)

# ### Draw histogram of dist
# fig, ax = plt.subplots(3,3)
# n_bins = 15
# ind = 0
# for i in range(3):
#     for j in range(3):
#         ax[i][j].hist(norm[:,ind], n_bins, alpha=0.5, density=True, label='lm_5')  
#         ax[i][j].hist(norm_1[:,ind], n_bins, alpha=0.5, density=True, label='lm_6') 
#         if i==2 and j==2:
#             ax[i][j].title.set_text('centroid')
#         else: 
#             ax[i][j].title.set_text('dist: keypoint_'+str(ind))
#         ax[i][j].legend(loc='right',prop={'size': 6})
#         ind+=1

# fig.tight_layout()
# dist_fig_hist = os.path.join(path, 'dist_hist_onlyMask.png')
# plt.savefig(dist_fig_hist)
# plt.clf()



### Draw each keypoint and centroid distance(from predicted to gt) VS add scatter correlation
fig, ax = plt.subplots(3,3)
ind = 0
for i in range(3):
    for j in range(3):
        ax[i][j].scatter(norm[:,ind], add, alpha=0.2, label='lm_5')
        ax[i][j].scatter(norm_1[:,ind], add_1, alpha=0.2, label='lm_6')
        #ax[i][j].legend(loc='upper left')
        if i==2 and j==2:
            ax[i][j].set_title('centroid'+' - add',fontsize=6)
        else: 
            ax[i][j].set_title('dist(keypoint_'+str(ind)+') - add',fontsize=6)
        ax[i][j].legend(loc='upper left',prop={'size': 6})
        ax[i][j].set_xlabel('dist',fontsize=6)
        ax[i][j].set_ylabel('add',fontsize=6)
        ind+=1

fig.tight_layout()
dist_fig = os.path.join(path, 'dist_add_onlyMask.png')
plt.savefig(dist_fig)
plt.clf()


### Draw each keypoint and centroid distance(from predicted to gt) VS loss_all scatter correlation
# fig, ax = plt.subplots(3,3)
# ind = 0
# for i in range(3):
#     for j in range(3):
#         ax[i][j].scatter(norm[:,ind], loss_all, alpha=0.2, label='RGB')
#         ax[i][j].scatter(norm_1[:,ind], loss_all_1, alpha=0.2, label='ours')
#         #ax[i][j].legend(loc='upper left')
#         if i==2 and j==2:
#             ax[i][j].title.set_text('centroid'+' - loss_all')
#         else: 
#             ax[i][j].title.set_text('dist(keypoint_'+str(ind)+' - loss_all')
#         ax[i][j].set_xlabel('dist')
#         ax[i][j].set_ylabel('loss_all')
#         ind+=1

# fig.tight_layout()
# dist_loss_fig = os.path.join(path, 'dist_loss_onlyMask.png')
# plt.savefig(dist_loss_fig)
# plt.clf()