import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import models
import math
import numpy as np
import torchvision.transforms as transforms
import PIL
import os
import matplotlib.pyplot as plt
from networks.resnet import ResNet, Bottleneck, BasicBlock_Res
from networks.vgg import vgg16
from collections import OrderedDict


class VGG_Feature(nn.Module):
    def __init__(self, fixed_param):
        super(VGG_Feature, self).__init__()

        self.fe = vgg16(pretrained=False)

        self.fe.load_state_dict(
            torch.load('networks/vgg16-397923af.pth'))

        features = self.fe.features

        self.to_feat = nn.Sequential()

        for i in range(15):
            self.to_feat.add_module(str(i), features[i])

        if fixed_param:
            for p in self.to_feat.parameters():
                p.requires_grad = False

    def forward(self, x):
        feature = self.to_feat(x)

        # feature = F.interpolate(feature, scale_factor=0.5, mode='bilinear', align_corners=True)

        return feature


class VGG_Bn_Feature(nn.Module):
    def __init__(self):
        super(VGG_Bn_Feature, self).__init__()

        features = models.vgg16_bn(pretrained=True).cuda().eval().features
        self.to_feat = nn.Sequential()
        # for i in range(8):
        #     self.to_feat.add_module(str(i), features[i])

        for i in range(15):
            self.to_feat.add_module(str(i), features[i])

        for p in self.to_feat.parameters():
            p.requires_grad = False

    def forward(self, x):
        feature = self.to_feat(x)

        # feature = F.interpolate(feature, scale_factor=0.5, mode='bilinear', align_corners=True)

        return feature


class Res18(nn.Module):
    def __init__(self):
        super(Res18, self).__init__()

        self.fe = ResNet(BasicBlock_Res, [2, 2, 2, 2])

        # self.fe = ResNet(Bottleneck, [3, 4, 6, 3])

        for p in self.fe.parameters():
            p.requires_grad = False

        self.fe.load_state_dict(
            torch.load('networks/resnet18-5c106cde.pth'))

    def forward(self, x):

        self.fe.eval()

        with torch.no_grad():
            feature = self.fe(x)

        return feature


class Res50(nn.Module):
    def __init__(self):
        super(Res50, self).__init__()

        self.fe = ResNet(Bottleneck, [3, 4, 6, 3])

        for p in self.fe.parameters():
            p.requires_grad = False

        # self.fe.load_state_dict(
        #     torch.load('networks/resnet50-19c8e357.pth'))
        self.fe.load_state_dict(
            torch.load('networks/DenseCL_R50_imagenet.pth'))

    def forward(self, x):

        self.fe.eval()

        with torch.no_grad():
            feature = self.fe(x)

        return feature


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    from collections import OrderedDict
    ckpt = torch.load('selfTrainVGG_withDA.pth')
    new_dict = OrderedDict()
    for k, v in ckpt.items():
        new_k = k.replace('module.', '')
        new_dict[new_k] = v

    torch.save(new_dict, 'selfTrainVGG_withDA.pth')