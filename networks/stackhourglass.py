from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from networks.submodule import convbn, convbn_3d, DisparityRegression


class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes*2)) #+conv2

        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes)) #+x

    def forward(self, x ,presqu, postsqu):

        out  = self.conv1(x) #in:1/4 out:1/8
        pre  = self.conv2(out) #in:1/8 out:1/8

        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        # print('pre2', pre.size())

        out  = self.conv3(pre) #in:1/8 out:1/16
        out  = self.conv4(out) #in:1/16 out:1/16

        # print('out', out.size())

        if presqu is not None:
            post = F.relu(self.conv5(out)+presqu, inplace=True) #in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out)+pre, inplace=True)

        out  = self.conv6(post)  #in:1/8 out:1/4

        return out, pre, post


class hourglass_gwcnet(nn.Module):
    def __init__(self, inplanes):
        super(hourglass_gwcnet, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 4, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(convbn_3d(inplanes * 4, inplanes * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes * 4, inplanes * 2, kernel_size=3, padding=1,
                                                      output_padding=1, stride=2, bias=False),
                                   nn.BatchNorm3d(inplanes * 2))
        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1,
                                                      output_padding=1, stride=2, bias=False),
                                   nn.BatchNorm3d(inplanes))

        self.redir1 = convbn_3d(inplanes, inplanes, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6

