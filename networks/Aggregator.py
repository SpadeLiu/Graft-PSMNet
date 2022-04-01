import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from networks.submodule import convbn, convbn_3d, DisparityRegression
from networks.stackhourglass import hourglass_gwcnet, hourglass
import matplotlib.pyplot as plt
import loss_functions as lf


def build_cost_volume(left_fea, right_fea, max_disp, cost_type):
    if cost_type == 'cor':

        left_fea_norm = F.normalize(left_fea, dim=1)
        right_fea_norm = F.normalize(right_fea, dim=1)

        cost = torch.zeros(left_fea.size()[0], 1, max_disp // 4,
                           left_fea.size()[2], left_fea.size()[3]).cuda()

        for i in range(max_disp // 4):
            if i > 0:
                cost[:, :, i, :, i:] = (torch.sum(left_fea_norm[:, :, :, i:] * right_fea_norm[:, :, :, :-i],
                                                  dim=1, keepdim=True) + 1) / 2
            else:
                cost[:, :, i, :, :] = (torch.sum(left_fea_norm * right_fea_norm, dim=1, keepdim=True) + 1) / 2

    elif cost_type == 'l2':
        cost = torch.zeros(left_fea.size()[0], 1, max_disp // 4,
                           left_fea.size()[2], left_fea.size()[3]).cuda()

        for i in range(max_disp // 4):
            if i > 0:
                cost[:, :, i, :, i:] = torch.sqrt(torch.sum(
                    (left_fea[:, :, :, i:] - right_fea[:, :, :, :-i]) ** 2, dim=1, keepdim=True))

            else:
                cost[:, :, i, :, :] = torch.sqrt(torch.sum((left_fea - right_fea) ** 2, dim=1, keepdim=True))

    elif cost_type == 'cat':

        cost = torch.zeros(left_fea.size()[0], left_fea.size()[1] * 2, max_disp // 4,
                           left_fea.size()[2], left_fea.size()[3]).cuda()

        for i in range(max_disp // 4):
            if i > 0:
                cost[:, :left_fea.size()[1], i, :, i:] = left_fea[:, :, :, i:]
                cost[:, left_fea.size()[1]:, i, :, i:] = right_fea[:, :, :, :-i]
            else:
                cost[:, :left_fea.size()[1], i, :, :] = left_fea
                cost[:, left_fea.size()[1]:, i, :, :] = right_fea

    elif cost_type == 'ncat':

        left_fea = F.normalize(left_fea, dim=1)
        right_fea = F.normalize(right_fea, dim=1)

        cost = torch.zeros(left_fea.size()[0], left_fea.size()[1] * 2, max_disp // 4,
                           left_fea.size()[2], left_fea.size()[3]).cuda()

        for i in range(max_disp // 4):
            if i > 0:
                cost[:, :left_fea.size()[1], i, :, i:] = left_fea[:, :, :, i:]
                cost[:, left_fea.size()[1]:, i, :, i:] = right_fea[:, :, :, :-i]
            else:
                cost[:, :left_fea.size()[1], i, :, :] = left_fea
                cost[:, left_fea.size()[1]:, i, :, :] = right_fea

    cost = cost.contiguous()

    return cost


class GwcAggregator(nn.Module):
    def __init__(self, maxdisp):
        super(GwcAggregator, self).__init__()
        self.maxdisp = maxdisp

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.hg1 = hourglass_gwcnet(32)
        self.hg2 = hourglass_gwcnet(32)
        self.hg3 = hourglass_gwcnet(32)

        self.classify1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        self.classify2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        self.classify3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left_fea, right_fea, gt_left, gt_right):
        cost = build_cost_volume(left_fea, right_fea, self.maxdisp, cost_type='ncat')

        cost0 = self.dres0(cost)
        cost1 = self.dres1(cost0) + cost0

        out1 = self.hg1(cost1)
        out2 = self.hg2(out1)
        out3 = self.hg3(out2)

        win_s = 5

        if self.training:
            cost1 = self.classify1(out1)
            cost1 = F.interpolate(cost1, scale_factor=4, mode='trilinear', align_corners=True)
            cost1 = torch.squeeze(cost1, 1)
            distribute1 = F.softmax(cost1, dim=1)
            pred1 = DisparityRegression(self.maxdisp, win_size=win_s)(distribute1)

            cost2 = self.classify2(out2)
            cost2 = F.interpolate(cost2, scale_factor=4, mode='trilinear', align_corners=True)
            cost2 = torch.squeeze(cost2, 1)
            distribute2 = F.softmax(cost2, dim=1)
            pred2 = DisparityRegression(self.maxdisp, win_size=win_s)(distribute2)

        cost3 = self.classify3(out3)
        cost3 = F.interpolate(cost3, scale_factor=4, mode='trilinear', align_corners=True)
        cost3 = torch.squeeze(cost3, 1)
        distribute3 = F.softmax(cost3, dim=1)
        pred3 = DisparityRegression(self.maxdisp, win_size=win_s)(distribute3)

        if self.training:
            mask = (gt_left < self.maxdisp) & (gt_left > 0)
            loss1 = 0.5 * F.smooth_l1_loss(pred1[mask], gt_left[mask]) + \
                    0.7 * F.smooth_l1_loss(pred2[mask], gt_left[mask]) + \
                    F.smooth_l1_loss(pred3[mask], gt_left[mask])

            gt_distribute = lf.disp2distribute(gt_left, self.maxdisp, b=2)
            loss2 = 0.5 * lf.CEloss(gt_left, self.maxdisp, gt_distribute, distribute1) + \
                    0.7 * lf.CEloss(gt_left, self.maxdisp, gt_distribute, distribute2) + \
                    lf.CEloss(gt_left, self.maxdisp, gt_distribute, distribute3)

            loss3 = lf.FeatureSimilarityLoss(self.maxdisp)(left_fea, right_fea, gt_left, gt_right)

            return loss1, loss2, loss3

        else:
            return pred3


class PSMAggregator(nn.Module):
    def __init__(self, maxdisp, udc):
        super(PSMAggregator, self).__init__()
        self.maxdisp = maxdisp
        self.udc = udc

        self.dres0 = nn.Sequential(convbn_3d(1, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.hg1 = hourglass(32)
        self.hg2 = hourglass(32)
        self.hg3 = hourglass(32)

        self.classify1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        self.classify2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        self.classify3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left_fea, right_fea, gt_left, training):
        cost = build_cost_volume(left_fea, right_fea, self.maxdisp, cost_type='cor')

        cost0 = self.dres0(cost)
        cost1 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.hg1(cost1, None, None)
        out1 = out1+cost0

        out2, pre2, post2 = self.hg2(out1, pre1, post1)
        out2 = out2+cost0

        out3, pre3, post3 = self.hg3(out2, pre1, post2)
        out3 = out3+cost0

        cost1 = self.classify1(out1)
        cost2 = self.classify2(out2) + cost1
        cost3 = self.classify3(out3) + cost2

        if self.udc:
            win_s = 5
        else:
            win_s = 0

        if self.training:
            cost1 = F.interpolate(cost1, scale_factor=4, mode='trilinear', align_corners=True)
            cost1 = torch.squeeze(cost1, 1)
            distribute1 = F.softmax(cost1, dim=1)
            pred1 = DisparityRegression(self.maxdisp, win_size=win_s)(distribute1)

            cost2 = F.interpolate(cost2, scale_factor=4, mode='trilinear', align_corners=True)
            cost2 = torch.squeeze(cost2, 1)
            distribute2 = F.softmax(cost2, dim=1)
            pred2 = DisparityRegression(self.maxdisp, win_size=win_s)(distribute2)

        cost3 = F.interpolate(cost3, scale_factor=4, mode='trilinear', align_corners=True)
        cost3 = torch.squeeze(cost3, 1)
        distribute3 = F.softmax(cost3, dim=1)
        pred3 = DisparityRegression(self.maxdisp, win_size=win_s)(distribute3)

        if self.training:
            mask = (gt_left < self.maxdisp) & (gt_left > 0)

            loss1 = 0.5 * F.smooth_l1_loss(pred1[mask], gt_left[mask]) + \
                    0.7 * F.smooth_l1_loss(pred2[mask], gt_left[mask]) + \
                    F.smooth_l1_loss(pred3[mask], gt_left[mask])

            gt_distribute = lf.disp2distribute(gt_left, self.maxdisp, b=2)
            loss2 = 0.5 * lf.CEloss(gt_left, self.maxdisp, gt_distribute, distribute1) + \
                    0.7 * lf.CEloss(gt_left, self.maxdisp, gt_distribute, distribute2) + \
                    lf.CEloss(gt_left, self.maxdisp, gt_distribute, distribute3)
            return loss1, loss2

        else:
            if training:
                mask = (gt_left < self.maxdisp) & (gt_left > 0)
                loss1 = F.smooth_l1_loss(pred3[mask], gt_left[mask])
                # loss2 = loss1
                gt_distribute = lf.disp2distribute(gt_left, self.maxdisp, b=2)
                loss2 = lf.CEloss(gt_left, self.maxdisp, gt_distribute, distribute3)
                return loss1, loss2

            else:
                return pred3
