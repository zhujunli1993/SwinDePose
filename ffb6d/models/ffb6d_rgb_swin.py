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

# psp_models = {
#     'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
#     'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
#     'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
# }


class FFB6D(nn.Module):
    def __init__(
        self, n_classes, n_pts, rndla_cfg, n_kps=8
    ):
        super().__init__()

        # ######################## prepare stages#########################
        self.n_cls = n_classes
        self.n_pts = n_pts
        self.n_kps = n_kps
        
        self.swin_ffb = swin.SwinTransformer(patch_size=4)
        self.psp_head = uper_head.UPerHead(in_channels=[96,192,384,768],channels=256,in_index=[0,1,2,3],num_classes=2)
        self.psp_fuse_head = uper_head.UPerHead(in_channels=[192,384,768,1536],channels=256,in_index=[0,1,2,3],num_classes=2)
        self.ds_rgb_swin = [384,384,384,384]
        self.ds_pc_swin = [64,64,64,64]
        self.ds_fuse_swin = [64,64,64,64]


        # ####################### prediction headers #############################
        # We use 3D keypoint prediction header for pose estimation following PVN3D
        # You can use different prediction headers for different downstream tasks.

        self.rgbd_seg_layer = (
            pt_utils.Seq(256*3)
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(n_classes, activation=None)
        )

        self.ctr_ofst_layer = (
            pt_utils.Seq(256*3)
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(3, activation=None)
        )

        self.kp_ofst_layer = (
            pt_utils.Seq(256*3)
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
        
        
        bs, _, h, w = inputs['rgb'].shape
        feat_nrm = self.swin_ffb(inputs['rgb']) # [1,96,120,160]->[1,192,60,80]->[1,384,30,40]->[1,768,15,20]
        feat_xyz = self.swin_ffb(inputs['dpt_xyz_map'].permute(0, 3, 1, 2)) # [1,96,120,160]->[1,192,60,80]->[1,384,30,40]->[1,768,15,20]
        # out = self.psp_head(feat)
        # ###################### encoding stages #############################
        ds_emb = []

        for i_ds in range(4):
            rgb_emb0 = feat_nrm[i_ds]
            p_emb0 = feat_xyz[i_ds]    
            # concat img and point feauture as global features
            ds_emb.append(torch.cat((rgb_emb0, p_emb0), dim=1))
        
        feat_up_nrm = self.psp_head(feat_nrm)
        feat_up_xyz = self.psp_head(feat_xyz)
        feat_up_dsEmb = self.psp_fuse_head(ds_emb)
        feat_up_fuse = torch.cat([feat_up_nrm, feat_up_xyz, feat_up_dsEmb], dim=1)
        intep = ops.Upsample(size=[h,w],mode='bilinear',align_corners=False)
        feat_final = intep(feat_up_fuse)
        
        
        bs, di, _, _ = feat_final.size()
        feat_final = feat_final.view(bs, di, -1)
        choose_emb = inputs['choose'].repeat(1, di, 1)
        feat_final = torch.gather(feat_final, 2, choose_emb).contiguous()

        # Use DenseFusion in final layer, which will hurt performance due to overfitting
        # rgbd_emb = self.fusion_layer(rgb_emb, pcld_emb)


        
        
        # ###################### prediction stages #############################
        rgbd_segs = self.rgbd_seg_layer(feat_final)
        pred_kp_ofs = self.kp_ofst_layer(feat_final)
        pred_ctr_ofs = self.ctr_ofst_layer(feat_final)

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

