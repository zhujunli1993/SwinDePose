import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cnn.pspnet_pseudo_depth_lm10 import PSPNet
import models.pytorch_utils as pt_utils
from models.RandLA.RandLANet import Network as RandLANet
from config.options import BaseOptions

opt = BaseOptions().parse()

psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=opt.psp_size, deep_features_size=opt.deep_features_size, backend='resnet34'),
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
        cnn = psp_models['resnet34'.lower()]()
        
        rndla = RandLANet(rndla_cfg)

        self.cnn_pre_stages = nn.Sequential(
            cnn.feats.conv1,  # stride = 2, [bs, c, 240, 320]
            cnn.feats.bn1, cnn.feats.relu,
            cnn.feats.maxpool  # stride = 2, [bs, 64, 120, 160]
        )
        
        self.cnn_pre_stages_depth = nn.Sequential(
            cnn.feats.conv1_depth,  # stride = 2, [bs, c, 240, 320]
            cnn.feats.bn1, cnn.feats.relu,
            cnn.feats.maxpool  # stride = 2, [bs, 64, 120, 160]
        )
        self.rndla_pre_stages = rndla.fc0

        # ####################### downsample stages#######################
        self.cnn_ds_stages = nn.ModuleList([
            cnn.feats.layer1,    # stride = 1, [bs, 64, 120, 160]
            cnn.feats.layer2,    # stride = 2, [bs, 128, 60, 80]
            # stride = 1, [bs, 128, 60, 80]
            # nn.Sequential(cnn.feats.layer3, cnn.feats.layer4),
            cnn.feats.layer3,
            nn.Sequential(cnn.psp, cnn.drop_1)   # [bs, 1024, 60, 80]
        ])
        
        self.cnn_depth_ds_stages = nn.ModuleList([
            cnn.feats.layer1    # stride = 1, [bs, 64, 120, 160]
            # cnn.feats.layer2,    # stride = 2, [bs, 128, 60, 80]
            # # stride = 1, [bs, 128, 60, 80]
            # # nn.Sequential(cnn.feats.layer3, cnn.feats.layer4),
            # cnn.feats.layer3,
            # nn.Sequential(cnn.psp, cnn.drop_1)   # [bs, 1024, 60, 80]
        ])
        
        self.ds_sr = [4, 8, 8, 8]

        self.rndla_ds_stages = rndla.dilated_res_blocks
        self.ds_rgb_ori_oc_fuse = opt.ds_rgb_ori_oc_fuse # [64, 128, 256, 512]
        self.ds_rgb_oc = opt.ds_rgb_oc # [64, 128, 256, 512]
        self.ds_rndla_oc = [item * 2 for item in rndla_cfg.d_out]
        self.ds_fuse_r2p_pre_layers = nn.ModuleList()
        self.ds_fuse_r2p_fuse_layers = nn.ModuleList()
        self.ds_fuse_p2r_pre_layers = nn.ModuleList()
        self.ds_fuse_p2r_fuse_layers = nn.ModuleList()
        # # Fuse the depth channel with the rgb features.
        self.ds_fuse_fd_fuse_layers = nn.ModuleList()
        self.ds_depth_oc_fuse = opt.ds_depth_oc_fuse #[128, 256, 512, 512]
        self.ds_depth_oc = opt.ds_depth_oc
        if opt.attention:
            for i in range(4):
                if i==0:
                    self.encoder_layer = nn.TransformerEncoderLayer(d_model=(self.ds_rgb_oc[i]+self.ds_depth_oc[i]), nhead=1)    
                    self.ds_fuse_fd_fuse_layers.append(nn.TransformerEncoder(self.encoder_layer, num_layers=1))
                else:
                    self.encoder_layer = nn.TransformerEncoderLayer(d_model=(self.ds_rgb_oc[i]), nhead=1)    
                    self.ds_fuse_fd_fuse_layers.append(nn.TransformerEncoder(self.encoder_layer, num_layers=1))
                if i==0:
                    self.ds_fuse_r2p_pre_layers.append(
                    pt_utils.Conv2d(
                        self.ds_rgb_oc[i]+self.ds_depth_oc[i], self.ds_rndla_oc[i], kernel_size=(1, 1),
                        bn=True
                        )
                    )      
                else:
                    self.ds_fuse_r2p_pre_layers.append(
                    pt_utils.Conv2d(
                        self.ds_rgb_oc[i], self.ds_rndla_oc[i], kernel_size=(1, 1),
                        bn=True
                        )
                    ) 
                self.ds_fuse_r2p_fuse_layers.append(
                pt_utils.Conv2d(
                    self.ds_rndla_oc[i]*2, self.ds_rndla_oc[i], kernel_size=(1, 1),
                    bn=True
                    )
                )

                self.ds_fuse_p2r_pre_layers.append(
                pt_utils.Conv2d(
                    self.ds_rndla_oc[i], self.ds_rgb_oc[i], kernel_size=(1, 1),
                    bn=True
                    )
                )
                if i==0:
                    self.ds_fuse_p2r_fuse_layers.append(
                    pt_utils.Conv2d(
                    self.ds_rgb_oc[i]+self.ds_depth_oc_fuse[i], self.ds_rgb_oc[i], kernel_size=(1, 1),
                    bn=True
                        )
                    )
                else:
                    self.ds_fuse_p2r_fuse_layers.append(
                    pt_utils.Conv2d(
                    self.ds_rgb_oc[i]*2, self.ds_rgb_oc[i], kernel_size=(1, 1),
                    bn=True
                        )
                    )
                
                
        else:    
            for i in range(4):
            
                self.ds_fuse_fd_fuse_layers.append(
                pt_utils.Conv2d(
                    self.ds_rgb_oc[i]+self.ds_depth_oc_fuse[i], self.ds_rgb_oc[i], kernel_size=(1, 1),
                    bn=True
                    )
                )

                self.ds_fuse_r2p_pre_layers.append(
                pt_utils.Conv2d(
                    self.ds_rgb_oc[i], self.ds_rndla_oc[i], kernel_size=(1, 1),
                    bn=True
                    )
                )
                self.ds_fuse_r2p_fuse_layers.append(
                    pt_utils.Conv2d(
                    self.ds_rndla_oc[i]*2, self.ds_rndla_oc[i], kernel_size=(1, 1),
                    bn=True
                    )
                )

                self.ds_fuse_p2r_pre_layers.append(
                    pt_utils.Conv2d(
                    self.ds_rndla_oc[i], self.ds_rgb_oc[i], kernel_size=(1, 1),
                    bn=True
                    )
                )
            
                self.ds_fuse_p2r_fuse_layers.append(
                    pt_utils.Conv2d(
                    self.ds_rgb_oc[i]*2, self.ds_rgb_oc[i], kernel_size=(1, 1),
                    bn=True
                    )
                )
            
        # ###################### upsample stages #############################
       
        
        self.up_rgb_oc = opt.up_rgb_oc # [256, 64, 64]
        self.cnn_up_stages = nn.ModuleList([
            nn.Sequential(cnn.up_1, cnn.drop_2),  # [bs, 256, 120, 160]
            nn.Sequential(cnn.up_2, cnn.drop_2),  # [bs, 64, 240, 320]
            nn.Sequential(cnn.final),  # [bs, 64, 240, 320]
            nn.Sequential(cnn.up_3, cnn.final)  # [bs, 64, 480, 640]
        ])
        
        self.up_rndla_oc = []
        for j in range(rndla_cfg.num_layers):
            if j < 3:
                self.up_rndla_oc.append(self.ds_rndla_oc[-j-2])
            else:
                self.up_rndla_oc.append(self.ds_rndla_oc[0])
        
        self.up_depth_oc= opt.up_depth_oc #[512, 256, 64,64]
        
        self.rndla_up_stages = rndla.decoder_blocks

        n_fuse_layer = 3
        self.up_fuse_r2p_pre_layers = nn.ModuleList()
        self.up_fuse_r2p_fuse_layers = nn.ModuleList()
        self.up_fuse_p2r_pre_layers = nn.ModuleList()
        self.up_fuse_p2r_fuse_layers = nn.ModuleList()
        self.up_fuse_fd_fuse_layers = nn.ModuleList()
        if opt.attention:
            
            for i in range(n_fuse_layer):
                
                
                self.up_fuse_r2p_pre_layers.append(
                    pt_utils.Conv2d(
                        self.up_rgb_oc[i], self.up_rndla_oc[i], kernel_size=(1, 1),
                        bn=True
                    )
                )
                self.up_fuse_r2p_fuse_layers.append(
                    pt_utils.Conv2d(
                        self.up_rndla_oc[i]*2, self.up_rndla_oc[i], kernel_size=(1, 1),
                        bn=True
                    )
                )

                self.up_fuse_p2r_pre_layers.append(
                    pt_utils.Conv2d(
                        self.up_rndla_oc[i], self.up_rgb_oc[i], kernel_size=(1, 1),
                        bn=True
                    )
                )
                self.up_fuse_p2r_fuse_layers.append(
                    pt_utils.Conv2d(
                        self.up_rgb_oc[i]*2, self.up_rgb_oc[i], kernel_size=(1, 1),
                        bn=True
                    )
                )
        else:
            for i in range(n_fuse_layer):
                self.up_fuse_fd_fuse_layers.append(
                    pt_utils.Conv2d(
                        self.up_rgb_oc[i]+self.up_depth_oc[i], self.up_rgb_oc[i], kernel_size=(1, 1),
                        bn=True
                    )
                    )
                self.up_fuse_r2p_pre_layers.append(
                    pt_utils.Conv2d(
                        self.up_rgb_oc[i], self.up_rndla_oc[i], kernel_size=(1, 1),
                        bn=True
                    )
                )
                self.up_fuse_r2p_fuse_layers.append(
                    pt_utils.Conv2d(
                        self.up_rndla_oc[i]*2, self.up_rndla_oc[i], kernel_size=(1, 1),
                        bn=True
                    )
                )

                self.up_fuse_p2r_pre_layers.append(
                    pt_utils.Conv2d(
                        self.up_rndla_oc[i], self.up_rgb_oc[i], kernel_size=(1, 1),
                        bn=True
                    )
                )
                self.up_fuse_p2r_fuse_layers.append(
                    pt_utils.Conv2d(
                        self.up_rgb_oc[i]*2, self.up_rgb_oc[i], kernel_size=(1, 1),
                        bn=True
                    )
                )
        # ####################### prediction headers #############################
        # We use 3D keypoint prediction header for pose estimation following PVN3D
        # You can use different prediction headers for different downstream tasks.

        self.rgbd_seg_layer = (
            pt_utils.Seq(self.up_rndla_oc[-1] + self.up_rgb_oc[-1])
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(n_classes, activation=None)
        )

        self.ctr_ofst_layer = (
            pt_utils.Seq(self.up_rndla_oc[-1]+self.up_rgb_oc[-1])
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(3, activation=None)
        )

        self.kp_ofst_layer = (
            pt_utils.Seq(self.up_rndla_oc[-1]+self.up_rgb_oc[-1])
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
            # angles      : FloatTensor [bs, 3, h, w]
            # sign_angles : FloatTensor [bs, 3, h, w]
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
        # ResNet pre + layer1 + layer2
        # Dirctly concat pseudo-rgb and depth together
        # rgb_emb = self.cnn_pre_stages(torch.cat((inputs['rgb'],inputs['depth'].unsqueeze(dim=1)),dim=1))  

        
        pseudo_emb0 = self.cnn_pre_stages(inputs['rgb']) 
        depth_emb0 = self.cnn_pre_stages_depth(inputs['depth'].unsqueeze(dim=1)) 
        # rndla pre
        # xyz, p_emb = self._break_up_pc(inputs['cld_rgb_nrm'])
        p_emb = inputs['cld_rgb_nrm'] # cld, rgb_c_pt, nrm_pt: selected points
        p_emb = self.rndla_pre_stages(p_emb)
        p_emb = p_emb.unsqueeze(dim=3)  # Batch*channel*npoints*1

        # ###################### encoding stages #############################
        ds_emb = []
        
        for i_ds in range(4):
            if i_ds == 0:
                # encode rgb downsampled feature
                pseudo_emb0 = self.cnn_ds_stages[i_ds](pseudo_emb0)
                # encode depth downsampled feature
                depth_emb0 = self.cnn_depth_ds_stages[i_ds](depth_emb0)
                
            if opt.attention and i_ds==0:
                bs, c, hr, wr = pseudo_emb0.size()
                bs_, c_, hr_, wr_ = depth_emb0.size()
                pseudo_emb0 = torch.reshape(pseudo_emb0, [bs,c, hr*wr])
                depth_emb0 = torch.reshape(depth_emb0, [bs_, c_, hr_*wr_])
                concat_pd = torch.permute(torch.cat((pseudo_emb0,depth_emb0),dim=1),[0,2,1])
                pseudo_emb0 = self.ds_fuse_fd_fuse_layers[i_ds](concat_pd)
                
                # reshape rgb_emb0 and depth_emb0 for the following fusion.
                img_emb = torch.permute(pseudo_emb0, [0, 2, 1])
                img_emb = torch.reshape(img_emb, [bs, -1, hr, wr]) # [8 128 120 160]
                depth_emb0 = torch.reshape(depth_emb0, [bs, -1, hr_, wr_])
            
            if opt.attention and not i_ds==0:
                img_emb = self.cnn_ds_stages[i_ds](img_emb)
                bs, c, hr, wr = img_emb.size()
                img_emb = torch.reshape(img_emb, [bs,c, hr*wr])
                img_emb = torch.permute(img_emb,[0,2,1])
                img_emb = self.ds_fuse_fd_fuse_layers[i_ds](img_emb)
                # reshape rgb_emb0 and depth_emb0 for the following fusion.
                img_emb = torch.permute(img_emb, [0, 2, 1])
                img_emb = torch.reshape(img_emb, [bs, -1, hr, wr]) # [8 128 120 160]
                
            if not opt.attention and not i_ds ==0:
                # encode rgb downsampled feature
                pseudo_emb0 = self.cnn_ds_stages[i_ds](torch.cat((pseudo_emb0,rgb_emb),dim=1))
                # encode depth downsampled feature
                depth_emb0 = self.cnn_depth_ds_stages[i_ds](torch.cat((depth_emb0,rgb_emb),dim=1))
                # Concat rgb_emb with depth_emb
                img_emb = self.ds_fuse_fd_fuse_layers[i_ds](torch.cat((depth_emb0,pseudo_emb0),dim=1))
            if not opt.attention and i_ds==0:
                # Concat rgb_emb with depth_emb
                img_emb = self.ds_fuse_fd_fuse_layers[i_ds](torch.cat((depth_emb0,pseudo_emb0),dim=1))
            
            bs, c, hr, wr = img_emb.size()

            # encode point cloud downsampled feature
            f_encoder_i = self.rndla_ds_stages[i_ds](
                p_emb, inputs['cld_xyz%d' % i_ds], inputs['cld_nei_idx%d' % i_ds]
            )
            p_emb0 = self.random_sample(f_encoder_i, inputs['cld_sub_idx%d' % i_ds])
            if i_ds == 0:
                ds_emb.append(f_encoder_i)

            # fuse point feauture to rgb feature
            p2r_emb = self.ds_fuse_p2r_pre_layers[i_ds](p_emb0)
            p2r_emb = self.nearest_interpolation(
                p2r_emb, inputs['p2r_ds_nei_idx%d' % i_ds]
            )
            p2r_emb = p2r_emb.view(bs, -1, hr, wr)
            rgb_emb = self.ds_fuse_p2r_fuse_layers[i_ds](
                torch.cat((img_emb, p2r_emb), dim=1)
            )
            
            
            # fuse rgb feature to point feature
            r2p_emb = self.random_sample(
                img_emb.reshape(bs, c, hr*wr, 1), inputs['r2p_ds_nei_idx%d' % i_ds]
            ).view(bs, c, -1, 1)
            r2p_emb = self.ds_fuse_r2p_pre_layers[i_ds](r2p_emb)
            p_emb = self.ds_fuse_r2p_fuse_layers[i_ds](
                torch.cat((p_emb0, r2p_emb), dim=1)
            )
            ds_emb.append(p_emb)
        
        # ###################### decoding stages #############################
        n_up_layers = len(self.rndla_up_stages)
        for i_up in range(n_up_layers-1):    

            img_emb = self.cnn_up_stages[i_up](rgb_emb)
            
            bs, c, hr, wr = img_emb.size()

            # decode point cloud upsampled feature
            f_interp_i = self.nearest_interpolation(
                p_emb, inputs['cld_interp_idx%d' % (n_up_layers-i_up-1)]
            )
            f_decoder_i = self.rndla_up_stages[i_up](
                torch.cat([ds_emb[-i_up - 2], f_interp_i], dim=1)
            )
            p_emb0 = f_decoder_i

            # fuse point feauture to rgb feature
            p2r_emb = self.up_fuse_p2r_pre_layers[i_up](p_emb0)
            p2r_emb = self.nearest_interpolation(
                p2r_emb, inputs['p2r_up_nei_idx%d' % i_up]
            )
            p2r_emb = p2r_emb.view(bs, -1, hr, wr)
            rgb_emb = self.up_fuse_p2r_fuse_layers[i_up](
                torch.cat((img_emb, p2r_emb), dim=1)
            )

            # fuse rgb feature to point feature
            r2p_emb = self.random_sample(
                img_emb.reshape(bs, c, hr*wr), inputs['r2p_up_nei_idx%d' % i_up]
            ).view(bs, c, -1, 1)
            r2p_emb = self.up_fuse_r2p_pre_layers[i_up](r2p_emb)
            p_emb = self.up_fuse_r2p_fuse_layers[i_up](
                torch.cat((p_emb0, r2p_emb), dim=1)
            )
           
        # final upsample layers:
        rgb_emb = self.cnn_up_stages[n_up_layers-1](rgb_emb)
        f_interp_i = self.nearest_interpolation(
            p_emb, inputs['cld_interp_idx%d' % (0)]
        )
        p_emb = self.rndla_up_stages[n_up_layers-1](
            torch.cat([ds_emb[0], f_interp_i], dim=1)
        ).squeeze(-1)

        bs, di, _, _ = rgb_emb.size()
        rgb_emb_c = rgb_emb.view(bs, di, -1)
        choose_emb = inputs['choose'].repeat(1, di, 1)
        rgb_emb_c = torch.gather(rgb_emb_c, 2, choose_emb).contiguous()

        # Use DenseFusion in final layer, which will hurt performance due to overfitting
        # rgbd_emb = self.fusion_layer(rgb_emb, pcld_emb)

        # Use simple concatenation. Good enough for fully fused RGBD feature.
        rgbd_emb = torch.cat([rgb_emb_c, p_emb], dim=1)

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

