import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad as Grad
from torchvision import transforms
import os
import copy
import skimage.io
from collections import OrderedDict
from tqdm import tqdm, trange
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse

from dataloader import ETH3D_loader as et
from dataloader.readpfm import readPFM
import networks.Aggregator as Agg
import networks.feature_extraction as FE
import networks.U_net as un


parser = argparse.ArgumentParser(description='GraftNet')
parser.add_argument('--no_cuda', action='store_true', default=False)
parser.add_argument('--gpu_id', type=str, default='2')
parser.add_argument('--seed', type=str, default=0)
parser.add_argument('--data_path', type=str, default='/media/data/dataset/ETH3D/')
parser.add_argument('--load_path', type=str, default='trained_models/checkpoint_final_10epoch.tar')
parser.add_argument('--max_disp', type=int, default=192)
args = parser.parse_args()


if not args.no_cuda:
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
cuda = torch.cuda.is_available()


all_limg, all_rimg, all_disp, all_mask = et.et_loader(args.data_path)


fe_model = FE.VGG_Feature(fixed_param=True).eval()
adaptor = un.U_Net_v4(img_ch=256, output_ch=64).eval()
agg_model = Agg.PSMAggregator(args.max_disp, udc=True).eval()

if cuda:
    fe_model = nn.DataParallel(fe_model.cuda())
    adaptor = nn.DataParallel(adaptor.cuda())
    agg_model = nn.DataParallel(agg_model.cuda())

adaptor.load_state_dict(torch.load(args.load_path)['fa_net'])
agg_model.load_state_dict(torch.load(args.load_path)['net'])


pred_mae = 0
pred_op = 0
for i in trange(len(all_limg)):
    limg = Image.open(all_limg[i]).convert('RGB')
    rimg = Image.open(all_rimg[i]).convert('RGB')

    w, h = limg.size
    wi, hi = (w // 16 + 1) * 16, (h // 16 + 1) * 16
    limg = limg.crop((w - wi, h - hi, w, h))
    rimg = rimg.crop((w - wi, h - hi, w, h))

    limg_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(limg)
    rimg_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(rimg)
    limg_tensor = limg_tensor.unsqueeze(0).cuda()
    rimg_tensor = rimg_tensor.unsqueeze(0).cuda()

    disp_gt, _ = readPFM(all_disp[i])
    disp_gt = np.ascontiguousarray(disp_gt, dtype=np.float32)
    disp_gt[disp_gt == np.inf] = 0
    gt_tensor = torch.FloatTensor(disp_gt).unsqueeze(0).unsqueeze(0).cuda()

    occ_mask = np.ascontiguousarray(Image.open(all_mask[i]))

    with torch.no_grad():
        left_fea = fe_model(limg_tensor)
        right_fea = fe_model(rimg_tensor)

        left_fea = adaptor(left_fea)
        right_fea = adaptor(right_fea)

        pred_disp = agg_model(left_fea, right_fea, gt_tensor, training=False)
        pred_disp = pred_disp[:, hi - h:, wi - w:]

    predict_np = pred_disp.squeeze().cpu().numpy()

    op_thresh = 1
    mask = (disp_gt > 0) & (occ_mask == 255)
    # mask = disp_gt > 0
    error = np.abs(predict_np * mask.astype(np.float32) - disp_gt * mask.astype(np.float32))

    pred_error = np.abs(predict_np * mask.astype(np.float32) - disp_gt * mask.astype(np.float32))
    pred_op += np.sum(pred_error > op_thresh) / np.sum(mask)
    pred_mae += np.mean(pred_error[mask])

print(pred_mae / len(all_limg))
print(pred_op / len(all_limg))