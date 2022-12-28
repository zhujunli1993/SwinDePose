import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cnn.pspnet_nrmOnly import PSPNet
import models.pytorch_utils as pt_utils
from models.RandLA.RandLANet import Network as RandLANet
# import swin_transformer as swin
from mmsegmentation.mmseg.models.backbones import swin
from mmsegmentation.mmseg.models.decode_heads import uper_head
from mmsegmentation.mmseg import ops

psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
}


class FFB6D(nn.Module):
    def __init__(
        self, n_classes, n_pts, rndla_cfg, n_kps=8
    ):
        super().__init__()

        # ######################## prepare stages#########################
        self.n_cls = n_classes
        self.n_pts = n_pts
        self.n_kps = n_kps
        self.swin_ffb = swin.SwinTransformer()
        self.psp_head = uper_head.UPerHead(in_channels=[96,192,384,768],channels=256,in_index=[0,1,2,3],num_classes=2)
        # self.psp_fuse_head = uper_head.UPerHead(in_channels=[192,384,768,1536],channels=256,in_index=[0,1,2,3],num_classes=2)
        self.ds_rgb_swin = [96,192,384,768]

        rndla = RandLANet(rndla_cfg)
        self.rndla_pre_stages = rndla.fc0

        # ####################### downsample stages#######################
        # self.cnn_ds_stages = nn.ModuleList([
        #     cnn.feats.layer1,    # stride = 1, [bs, 64, 120, 160]
        #     cnn.feats.layer2,    # stride = 2, [bs, 128, 60, 80]
        #     # stride = 1, [bs, 128, 60, 80]
        #     nn.Sequential(cnn.feats.layer3, cnn.feats.layer4),
        #     nn.Sequential(cnn.psp, cnn.drop_1)   # [bs, 1024, 60, 80]
        # ])
        self.ds_sr = [4, 8, 8, 8]
        self.rndla_ds_stages = rndla.dilated_res_blocks
        self.ds_rndla_oc = [item * 2 for item in rndla_cfg.d_out]
        
        # self.ds_fuse_rp_fuse_layers = nn.ModuleList()
        # for i in range(4):
        #     self.ds_fuse_rp_fuse_layers.append(
        #         pt_utils.Conv2d(
        #             self.ds_rgb_swin[i]+self.ds_rndla_oc[i], kernel_size=(1, 1),
        #             bn=True
        #         )
        #     )
        
        self.fuse_emb = [192,384,768,1536]    

        # ###################### upsample stages #############################
        # self.cnn_up_stages = nn.ModuleList([
        #     nn.Sequential(cnn.up_1, cnn.drop_2),  # [bs, 256, 120, 160]
        #     nn.Sequential(cnn.up_2, cnn.drop_2),  # [bs, 64, 240, 320]
        #     nn.Sequential(cnn.final),  # [bs, 64, 240, 320]
        #     nn.Sequential(cnn.up_3, cnn.final)  # [bs, 64, 480, 640]
        # ])
        # self.up_rgb_oc = [256, 64, 64]
        self.up_rndla_oc = []
        for j in range(rndla_cfg.num_layers):
            if j < 3:
                self.up_rndla_oc.append(self.ds_rndla_oc[-j-2])
            else:
                self.up_rndla_oc.append(self.ds_rndla_oc[0])

        self.rndla_up_stages = rndla.decoder_blocks

        # n_fuse_layer = 3
        # self.up_fuse_r2p_pre_layers = nn.ModuleList()
        # self.up_fuse_r2p_fuse_layers = nn.ModuleList()
        # self.up_fuse_p2r_pre_layers = nn.ModuleList()
        # self.up_fuse_p2r_fuse_layers = nn.ModuleList()
        # for i in range(n_fuse_layer):
        #     self.up_fuse_r2p_pre_layers.append(
        #         pt_utils.Conv2d(
        #             self.up_rgb_oc[i], self.up_rndla_oc[i], kernel_size=(1, 1),
        #             bn=True
        #         )
        #     )
        #     self.up_fuse_r2p_fuse_layers.append(
        #         pt_utils.Conv2d(
        #             self.up_rndla_oc[i]*2, self.up_rndla_oc[i], kernel_size=(1, 1),
        #             bn=True
        #         )
        #     )

        #     self.up_fuse_p2r_pre_layers.append(
        #         pt_utils.Conv2d(
        #             self.up_rndla_oc[i], self.up_rgb_oc[i], kernel_size=(1, 1),
        #             bn=True
        #         )
        #     )
        #     self.up_fuse_p2r_fuse_layers.append(
        #         pt_utils.Conv2d(
        #             self.up_rgb_oc[i]*2, self.up_rgb_oc[i], kernel_size=(1, 1),
        #             bn=True
        #         )
        #     )

        # ####################### prediction headers #############################
        # We use 3D keypoint prediction header for pose estimation following PVN3D
        # You can use different prediction headers for different downstream tasks.

        self.rgbd_seg_layer = (
            pt_utils.Seq(320)
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(n_classes, activation=None)
        )

        self.ctr_ofst_layer = (
            pt_utils.Seq(320)
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(3, activation=None)
        )

        self.kp_ofst_layer = (
            pt_utils.Seq(320)
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(n_kps*3, activation=None)
        )

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        if len(feature.size()) > 3:
            feature = feature.squeeze(dim=3)  # batch*channel*npoints
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(
            feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1)
        ).contiguous()
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]  # batch*channel*npoints*1
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        interpolated_features = torch.gather(
            feature, 2, interp_idx.unsqueeze(1).repeat(1, feature.shape[1], 1)
        ).contiguous()
        interpolated_features = interpolated_features.unsqueeze(3)  # batch*channel*npoints*1
        return interpolated_features

    def _break_up_pc(self, pc):
        xyz = pc[:, :3, :].transpose(1, 2).contiguous()
        features = (
            pc[:, 3:, :].contiguous() if pc.size(1) > 3 else None
        )
        return xyz, features

    def forward(
        self, inputs, end_points=None, scale=1,
    ):
        """
        Params:
        inputs: dict of :
            rgb         : FloatTensor [bs, 3, h, w]
            dpt_nrm     : FloatTensor [bs, 6, h, w], 3c xyz in meter + 3c normal map
            cld_rgb_nrm : FloatTensor [bs, 9, npts]
            choose      : LongTensor [bs, 1, npts]
            xmap, ymap: [bs, h, w]
            K:          [bs, 3, 3]
        Returns:
            end_points:
        """
        # ###################### prepare stages #############################
        if not end_points:
            end_points = {}
        
        bs, _, h, w = inputs['nrm_angles'].shape
        feat_nrm = self.swin_ffb(inputs['nrm_angles']) # [1,96,120,160]->[1,192,60,80]->[1,384,30,40]->[1,768,15,20]
        p_emb = inputs['cld_angle_nrm']
        p_emb = self.rndla_pre_stages(p_emb) #[1, 8, 19200, 1]
        p_emb = p_emb.unsqueeze(dim=3)  # Batch*channel*npoints*1
        ds_pc_emb = []
        
        for i_ds in range(4):
            # encode point cloud downsampled feature
            f_encoder_i = self.rndla_ds_stages[i_ds](
                p_emb, inputs['cld_xyz%d' % i_ds], inputs['cld_nei_idx%d' % i_ds]
            ) # f_encoder_i.shape=[1, 64, 19200, 1] -> [1, 128, 4800, 1] -> [1, 256, 1200, 1] -> [1, 512, 300, 1]
            
            p_emb0 = self.random_sample(f_encoder_i, inputs['cld_sub_idx%d' % i_ds]) # [1, 64, 4800, 1] -> [1, 128, 1200, 1] ->[1, 256, 300, 1] -> [1, 512, 75, 1]
            if i_ds == 0:
                ds_pc_emb.append(f_encoder_i)
            p_emb = p_emb0
            ds_pc_emb.append(p_emb)
        # ds_pc_emb: [1, 64, 19200, 1] -> [1, 64, 4800, 1] -> [1, 128, 1200, 1] -> [1, 256, 300, 1] -> [1, 512, 75, 1]
        
        # ###################### decoding stages #############################
        
        feat_up_nrm = self.psp_head(feat_nrm)
        n_up_layers = len(self.rndla_up_stages)
        for i_up in range(n_up_layers-1):
            # decode point cloud upsampled feature
            f_interp_i = self.nearest_interpolation(
                p_emb, inputs['cld_interp_idx%d' % (n_up_layers-i_up-1)]
            )# f_interp_i: [1, 512, 300, 1] -> [1, 256, 1200, 1] -> [1, 128, 4800, 1]
            f_decoder_i = self.rndla_up_stages[i_up](
                torch.cat([ds_pc_emb[-i_up - 2], f_interp_i], dim=1)
            ) # f_decoder_i: [1, 256, 300, 1] -> [1, 128, 1200, 1] -> [1, 64, 4800, 1]
            p_emb = f_decoder_i

        
        # final upsample layers:
        f_interp_i = self.nearest_interpolation(
            p_emb, inputs['cld_interp_idx%d' % (0)]
        ) # f_interp_i: [1, 64, 19200, 1]
        p_emb = self.rndla_up_stages[n_up_layers-1](
            torch.cat([ds_pc_emb[0], f_interp_i], dim=1)
        ).squeeze(-1) # p_emb: [1, 64, 19200]
        
        # # For data visualization
        # import pdb;pdb.set_trace()
        # id_ind = str(int(inputs['img_id'].cpu().numpy()[0]))
        
        # torch.save(inputs['cld_angle_nrm'], os.path.join('/workspace','REPO','pose_estimation','ffb6d','train_log','lm_swinTiny_phone_fullSyn_dense_fullInc','phone',id_ind+'_pc.pt'))
        # torch.save(feat_up_nrm, os.path.join('/workspace','REPO','pose_estimation','ffb6d','train_log','lm_swinTiny_phone_fullSyn_dense_fullInc','phone',id_ind+'_small_img.pt'))
        
        
        bs, di, _, _ = feat_up_nrm.size() # feat_up_nrm: [1, 256, 120, 160]
        # feat_up_nrm = feat_up_nrm.view(bs, di, -1)
        intep = ops.Upsample(size=[h,w],mode='bilinear',align_corners=False)
        feat_final_nrm = intep(feat_up_nrm)
        
        # # For data visualization
        
        # torch.save(feat_final_nrm, os.path.join('/workspace','REPO','pose_estimation','ffb6d','train_log','lm_swinTiny_phone_fullSyn_dense_fullInc','phone',id_ind+'_img.pt'))
        
        feat_final_nrm = feat_final_nrm.view(bs, di, -1)
        choose_emb = inputs['choose'].repeat(1, di, 1)
        nrm_emb_c = torch.gather(feat_final_nrm, 2, choose_emb).contiguous() # nrm_emb_c: [1, 256, 120, 160]

        # Use DenseFusion in final layer, which will hurt performance due to overfitting
        # rgbd_emb = self.fusion_layer(rgb_emb, pcld_emb)
        
        # Use simple concatenation. Good enough for fully fused RGBD feature.
        rgbd_emb = torch.cat([nrm_emb_c, p_emb], dim=1)
        
        
        # # # For data visualization
        # torch.save(rgbd_emb, os.path.join('/workspace','REPO','pose_estimation','ffb6d','train_log','lm_swinTiny_phone_fullSyn_dense_fullInc','phone',id_ind+'_img_pc.pt'))
        # torch.save(nrm_emb_c, os.path.join('/workspace','REPO','pose_estimation','ffb6d','train_log','lm_swinTiny_phone_fullSyn_dense_fullInc','phone',id_ind+'_img_final.pt'))
        # torch.save(p_emb, os.path.join('/workspace','REPO','pose_estimation','ffb6d','train_log','lm_swinTiny_phone_fullSyn_dense_fullInc','phone',id_ind+'_pc_final.pt'))
        
        # ###################### prediction stages #############################
        rgbd_segs = self.rgbd_seg_layer(rgbd_emb)
        pred_kp_ofs = self.kp_ofst_layer(rgbd_emb)
        pred_ctr_ofs = self.ctr_ofst_layer(rgbd_emb)

        pred_kp_ofs = pred_kp_ofs.view(
            bs, self.n_kps, 3, -1
        ).permute(0, 1, 3, 2).contiguous()
        pred_ctr_ofs = pred_ctr_ofs.view(
            bs, 1, 3, -1
        ).permute(0, 1, 3, 2).contiguous()

        # return rgbd_seg, pred_kp_of, pred_ctr_of
        end_points['pred_rgbd_segs'] = rgbd_segs
        end_points['pred_kp_ofs'] = pred_kp_ofs
        end_points['pred_ctr_ofs'] = pred_ctr_ofs

        
        
        return end_points


# Copy from PVN3D: https://github.com/ethnhe/PVN3D
class DenseFusion(nn.Module):
    def __init__(self, num_points):
        super(DenseFusion, self).__init__()
        self.conv2_rgb = torch.nn.Conv1d(64, 256, 1)
        self.conv2_cld = torch.nn.Conv1d(32, 256, 1)

        self.conv3 = torch.nn.Conv1d(96, 512, 1)
        self.conv4 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)

    def forward(self, rgb_emb, cld_emb):
        bs, _, n_pts = cld_emb.size()
        feat_1 = torch.cat((rgb_emb, cld_emb), dim=1)
        rgb = F.relu(self.conv2_rgb(rgb_emb))
        cld = F.relu(self.conv2_cld(cld_emb))

        feat_2 = torch.cat((rgb, cld), dim=1)

        rgbd = F.relu(self.conv3(feat_1))
        rgbd = F.relu(self.conv4(rgbd))

        ap_x = self.ap1(rgbd)

        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, n_pts)
        return torch.cat([feat_1, feat_2, ap_x], 1)  # 96+ 512 + 1024 = 1632

