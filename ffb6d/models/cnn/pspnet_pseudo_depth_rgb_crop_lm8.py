import torch
from torch import nn
from torch.nn import functional as F
import models.cnn.extractors_pseudo_depth_rgb_crop_lm8 as extractors_pseudo_depth_rgb
from config.options import BaseOptions
opt = BaseOptions().parse()

class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(features, size) for size in sizes]
        )
        self.bottleneck = nn.Conv2d(
            features * (len(sizes) + 1), out_features, kernel_size=1
        )
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [
            F.interpolate(input=stage(feats), size=(h, w), mode='bilinear')
            for stage in self.stages
        ] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PSPUpsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # F.interpolate(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv(x)


class Modified_PSPNet(nn.Module):
    def __init__(self, n_classes=22, sizes=(1, 2, 3, 6), psp_size=2048,
                 deep_features_size=1024, backend='resnet18', pretrained=True
                 ):
        super(Modified_PSPNet, self).__init__()
        self.feats = getattr(extractors_pseudo_depth_rgb, backend)(pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
        )

        self.final_seg = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.LogSoftmax()
        )

        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        f, class_f = self.feats(x)
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)

        return self.final(p), self.final_seg(p).permute(0, 2, 3, 1).contiguous()


class PSPNet(nn.Module):
    def __init__(
            self, n_classes=22, sizes=(1, 2, 3, 6), psp_size=2048,
            deep_features_size=1024, backend='resnet18', pretrained=False
    ):
        super(PSPNet, self).__init__()
        self.feats = getattr(extractors_pseudo_depth_rgb, backend)(pretrained)
        self.psp = PSPModule(psp_size, opt.psp_out, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(opt.psp_out, opt.up_rgb_oc[0])
        self.up_2 = PSPUpsample(opt.up_rgb_oc[0], opt.up_rgb_oc[1])
        self.up_3 = PSPUpsample(opt.up_rgb_oc[1], opt.up_rgb_oc[2])

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            # nn.Conv2d(64, 32, kernel_size=1),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.LogSoftmax()
        )

        self.final_seg = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.LogSoftmax()
        )

        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        f, class_f = self.feats(x)
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)

        return self.final(p), self.final_seg(p).permute(0, 2, 3, 1).contiguous()