from __future__ import print_function
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


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes))


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class DisparityRegression(nn.Module):

    def __init__(self, maxdisp, win_size):
        super(DisparityRegression, self).__init__()
        self.max_disp = maxdisp
        self.win_size = win_size

    def forward(self, x):
        disp = torch.arange(0, self.max_disp).view(1, -1, 1, 1).float().to(x.device)

        if self.win_size > 0:
            max_d = torch.argmax(x, dim=1, keepdim=True)
            d_value = []
            prob_value = []
            for d in range(-self.win_size, self.win_size + 1):
                index = max_d + d
                index[index < 0] = 0
                index[index > x.shape[1] - 1] = x.shape[1] - 1
                d_value.append(index)

                prob = torch.gather(x, dim=1, index=index)
                prob_value.append(prob)

            part_x = torch.cat(prob_value, dim=1)
            part_x = part_x / (torch.sum(part_x, dim=1, keepdim=True) + 1e-8)
            part_d = torch.cat(d_value, dim=1).float()
            out = torch.sum(part_x * part_d, dim=1)

        else:
            out = torch.sum(x * disp, 1)

        return out


class GwcFeature(nn.Module):
    def __init__(self, out_c, fuse_mode='add'):
        super(GwcFeature, self).__init__()
        self.inplanes = 32
        self.fuse_mode = fuse_mode

        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        if self.fuse_mode == 'cat':
            self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, out_c, kernel_size=1, padding=0, stride=1, bias=False))
        elif self.fuse_mode == 'add':
            self.l1_conv = nn.Conv2d(32, out_c, 1, stride=1, padding=0, bias=False)
            self.l2_conv = nn.Conv2d(64, out_c, 1, stride=1, padding=0, bias=False)
            self.l4_conv = nn.Conv2d(128, out_c, 1, stride=1, padding=0, bias=False)
        elif self.fuse_mode == 'add_sa':
            self.l1_conv = nn.Conv2d(64, out_c, 1, stride=1, padding=0, bias=False)
            self.l4_conv = nn.Conv2d(64, out_c, 1, stride=1, padding=0, bias=False)
            self.sa = nn.Sequential(convbn(2 * out_c, 2 * out_c, 3, 1, 1, 1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(2 * out_c, 2, 3, stride=1, padding=1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.firstconv(x)
        output_l1 = self.layer1(output)
        output_l2 = self.layer2(output_l1)
        output_l3 = self.layer3(output_l2)
        output_l4 = self.layer4(output_l3)

        output_l1 = F.interpolate(output_l1, (output_l4.size()[2], output_l4.size()[3]),
                                  mode='bilinear', align_corners=True)

        if self.fuse_mode == 'cat':
            cat_feature = torch.cat((output_l2, output_l3, output_l4), dim=1)
            output_feature = self.lastconv(cat_feature)
        elif self.fuse_mode == 'add':
            output_l1 = self.l1_conv(output_l1)
            output_l4 = self.l4_conv(output_l4)
            output_feature = output_l1 + output_l4
        elif self.fuse_mode == 'add_sa':
            output_l1 = self.l1_conv(output_l1)
            output_l4 = self.l4_conv(output_l4)

            attention_map = self.sa(torch.cat((output_l1, output_l4), dim=1))
            attention_map = torch.sigmoid(attention_map)
            output_feature = output_l1 * attention_map[:, 0].unsqueeze(1) + \
                             output_l4 * attention_map[:, 1].unsqueeze(1)

        return output_feature


