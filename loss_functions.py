import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


def disp2distribute(disp_gt, max_disp, b=2):
    disp_gt = disp_gt.unsqueeze(1)
    disp_range = torch.arange(0, max_disp).view(1, -1, 1, 1).float().cuda()
    gt_distribute = torch.exp(-torch.abs(disp_range - disp_gt) / b)
    gt_distribute = gt_distribute / (torch.sum(gt_distribute, dim=1, keepdim=True) + 1e-8)
    return gt_distribute


def CEloss(disp_gt, max_disp, gt_distribute, pred_distribute):
    mask = (disp_gt > 0) & (disp_gt < max_disp)

    pred_distribute = torch.log(pred_distribute + 1e-8)

    ce_loss = torch.sum(-gt_distribute * pred_distribute, dim=1)
    ce_loss = torch.mean(ce_loss[mask])
    return ce_loss


def gradient_x(img):
    img = F.pad(img, (0, 1, 0, 0), mode="replicate")
    gx = img[:, :, :, :-1] - img[:, :, :, 1:]
    return gx


def gradient_y(img):
    img = F.pad(img, (0, 0, 0, 1), mode="replicate")
    gy = img[:, :, :-1, :] - img[:, :, 1:, :]
    return gy


def smooth_loss(img, disp):
    img_gx = gradient_x(img)
    img_gy = gradient_y(img)
    disp_gx = gradient_x(disp)
    disp_gy = gradient_y(disp)

    weight_x = torch.exp(-torch.mean(torch.abs(img_gx), dim=1, keepdim=True))
    weight_y = torch.exp(-torch.mean(torch.abs(img_gy), dim=1, keepdim=True))
    smoothness_x = torch.abs(disp_gx * weight_x)
    smoothness_y = torch.abs(disp_gy * weight_y)
    smoothness_loss = smoothness_x + smoothness_y

    return torch.mean(smoothness_loss)



def occlusion_mask(left_disp, right_disp, threshold=1):
    # left_disp = left_disp.unsqueeze(1)
    # right_disp = right_disp.unsqueeze(1)

    B, _, H, W = left_disp.size()

    x_base = torch.linspace(0, 1, W).repeat(B, H, 1).type_as(right_disp)
    y_base = torch.linspace(0, 1, H).repeat(B, W, 1).transpose(1, 2).type_as(right_disp)

    flow_field = torch.stack((x_base - left_disp.squeeze(1) / W, y_base), dim=3)

    recon_left_disp = F.grid_sample(right_disp, 2 * flow_field - 1, mode='bilinear', padding_mode='zeros')

    lr_check = torch.abs(recon_left_disp - left_disp)
    mask = lr_check > threshold

    return mask


def reconstruction(right, disp):
    b, _, h, w = right.size()

    x_base = torch.linspace(0, 1, w).repeat(b, h, 1).type_as(right)
    y_base = torch.linspace(0, 1, h).repeat(b, w, 1).transpose(1, 2).type_as(right)

    flow_field = torch.stack((x_base - disp / w, y_base), dim=3)

    recon_left = F.grid_sample(right, 2 * flow_field - 1, mode='bilinear', padding_mode='zeros')
    return recon_left


def NT_Xent_loss(positive_simi, negative_simi, t):
    loss = torch.exp(positive_simi / t) / \
           (torch.exp(positive_simi / t) + torch.sum(torch.exp(negative_simi / t), dim=4))
    loss = -torch.log(loss + 1e-9)
    return loss


class FeatureSimilarityLoss(nn.Module):
    def __init__(self, max_disp):
        super(FeatureSimilarityLoss, self).__init__()
        self.max_disp = max_disp
        self.m = 0.3
        self.nega_num = 1

    def forward(self, left_fea, right_fea, left_disp, right_disp):
        B, _, H, W = left_fea.size()

        down_disp = F.interpolate(left_disp, (H, W), mode='nearest') / 4.
        # down_img = F.interpolate(left_img, (H, W), mode='nearest')
        # down_img = torch.mean(down_img, dim=1, keepdim=True)

        # t_map = self.t_net(left_fea)

        # create negative samples
        random_offset = torch.rand(B, self.nega_num, H, W).cuda() * 2 + 1
        random_sign = torch.sign(torch.rand(B, self.nega_num, H, W).cuda() - 0.5)
        random_offset *= random_sign
        negative_disp = down_disp + random_offset

        positive_recon = reconstruction(right_fea, down_disp.squeeze(1))
        negative_recon = []
        for i in range(self.nega_num):
            negative_recon.append(reconstruction(right_fea, negative_disp[:, i]))
        negative_recon = torch.stack(negative_recon, dim=4)

        left_fea = F.normalize(left_fea, dim=1)
        positive_recon = F.normalize(positive_recon, dim=1)
        negative_recon = F.normalize(negative_recon, dim=1)

        positive_simi = (torch.sum(left_fea * positive_recon, dim=1, keepdim=True) + 1) / 2
        negative_simi = (torch.sum(left_fea.unsqueeze(4) * negative_recon, dim=1, keepdim=True) + 1) / 2

        judge_mat_p = torch.zeros_like(positive_simi)
        judge_mat_n = torch.zeros_like(negative_simi)
        if torch.sum(positive_simi < judge_mat_p) > 0 or torch.sum(negative_simi < judge_mat_n) > 0:
            print('cosine_simi < 0')

        # hinge loss
        # dist = self.m + negative_simi - positive_simi
        # criteria = torch.zeros_like(dist)
        # loss, _ = torch.max(torch.cat((dist, criteria), dim=1), dim=1, keepdim=True)

        # NT-Xent loss
        # loss = NT_Xent_loss(positive_simi, negative_simi, t=t_map)
        loss = NT_Xent_loss(positive_simi, negative_simi, t=0.2)

        # img_grad = torch.sqrt(gradient_x(down_img) ** 2 + gradient_y(down_img) ** 2)
        # weight = torch.exp(-img_grad)
        # loss = loss * weight

        occ_mask = occlusion_mask(left_disp, right_disp, threshold=1)
        occ_mask = F.interpolate(occ_mask.float(), (H, W), mode='nearest')
        valid_mask = (down_disp > 0) & (down_disp < self.max_disp // 4) & (occ_mask == 0)

        return torch.mean(loss[valid_mask])


def gram_matrix(feature):
    B, C, H, W = feature.size()
    feature = feature.view(B, C, H * W)
    feature_t = feature.transpose(1, 2)
    gram_m = torch.bmm(feature, feature_t) / (H * W)
    return gram_m


def gram_matrix_v2(feature):
    B, C, H, W = feature.size()
    feature = feature.view(B * C, H * W)
    gram_m = torch.mm(feature, feature.t()) / (B * C * H * W)
    return gram_m


if __name__ == '__main__':

    a = torch.rand(2, 256, 64, 128)
    b = torch.rand(2, 256, 64, 128)

    gram_a = gram_matrix(a)
    gram_b = gram_matrix(b)
    print(F.mse_loss(gram_a, gram_b))

    ga_2 = gram_matrix_v2(a)
    gb_2 = gram_matrix_v2(b)
    print(F.mse_loss(ga_2, gb_2))
