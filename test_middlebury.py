import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
from torch.autograd import grad as Grad
import skimage.io
import os
import copy
from collections import OrderedDict
from tqdm import tqdm, trange
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse

from dataloader import middlebury_loader as mb
from dataloader import readpfm as rp
import networks.Aggregator as Agg
import networks.U_net as un
import networks.feature_extraction as FE


parser = argparse.ArgumentParser(description='GraftNet')
parser.add_argument('--no_cuda', action='store_true', default=False)
parser.add_argument('--gpu_id', type=str, default='2')
parser.add_argument('--seed', type=str, default=0)
parser.add_argument('--resolution', type=str, default='H')
parser.add_argument('--data_path', type=str, default='/media/data/dataset/MiddEval3-data-H/')
parser.add_argument('--load_path', type=str, default='trained_models/checkpoint_final_10epoch.tar')
parser.add_argument('--max_disp', type=int, default=192)
args = parser.parse_args()

if not args.no_cuda:
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
cuda = torch.cuda.is_available()

train_limg, train_rimg, train_gt, test_limg, test_rimg = mb.mb_loader(args.data_path, res=args.resolution)

fe_model = FE.VGG_Feature(fixed_param=True).eval()
adaptor = un.U_Net_v4(img_ch=256, output_ch=64).eval()
agg_model = Agg.PSMAggregator(args.max_disp, udc=True).eval()

if cuda:
    fe_model = nn.DataParallel(fe_model.cuda())
    adaptor = nn.DataParallel(adaptor.cuda())
    agg_model = nn.DataParallel(agg_model.cuda())

adaptor.load_state_dict(torch.load(args.load_path)['fa_net'])
agg_model.load_state_dict(torch.load(args.load_path)['net'])


def test_trainset():
    op = 0
    mae = 0

    for i in trange(len(train_limg)):

        limg_path = train_limg[i]
        rimg_path = train_rimg[i]

        limg = Image.open(limg_path).convert('RGB')
        rimg = Image.open(rimg_path).convert('RGB')

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

        with torch.no_grad():
            left_fea = fe_model(limg_tensor)
            right_fea = fe_model(rimg_tensor)

            left_fea = adaptor(left_fea)
            right_fea = adaptor(right_fea)

            pred_disp = agg_model(left_fea, right_fea, limg_tensor, training=False)
            pred_disp = pred_disp[:, hi - h:, wi - w:]

        pred_np = pred_disp.squeeze().cpu().numpy()

        torch.cuda.empty_cache()

        disp_gt, _ = rp.readPFM(train_gt[i])
        disp_gt = np.ascontiguousarray(disp_gt, dtype=np.float32)
        disp_gt[disp_gt == np.inf] = 0

        occ_mask = Image.open(train_gt[i].replace('disp0GT.pfm', 'mask0nocc.png')).convert('L')
        occ_mask = np.ascontiguousarray(occ_mask, dtype=np.float32)

        mask = (disp_gt <= 0) | (occ_mask != 255) | (disp_gt >= args.max_disp)
        # mask = (disp_gt <= 0) | (disp_gt >= maxdisp)

        error = np.abs(pred_np - disp_gt)
        error[mask] = 0

        if i in [6, 8, 9, 12, 14]:
            k = 1
        else:
            k = 1

        op += np.sum(error > 2.0) / (w * h - np.sum(mask)) * k
        mae += np.sum(error) / (w * h - np.sum(mask)) * k

    print(op / 15 * 100)
    print(mae / 15)


if __name__ == '__main__':
    test_trainset()
    # test_testset()