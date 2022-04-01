import torch
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import os
import copy
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import argparse

from dataloader import sceneflow_loader as sf
import networks.Aggregator as Agg
import networks.submodule as sm
import networks.U_net as un
import networks.feature_extraction as FE
import loss_functions as lf


parser = argparse.ArgumentParser(description='GraftNet')
parser.add_argument('--no_cuda', action='store_true', default=False)
parser.add_argument('--gpu_id', type=str, default='0, 1')
parser.add_argument('--seed', type=str, default=0)
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--data_path', type=str, default='/media/data/dataset/SceneFlow/')
parser.add_argument('--save_path', type=str, default='trained_models/')
parser.add_argument('--load_path', type=str, default='trained_models/checkpoint_adaptor_1epoch.tar')
parser.add_argument('--max_disp', type=int, default=192)
parser.add_argument('--color_transform', action='store_true', default=False)
args = parser.parse_args()

if not args.no_cuda:
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)
if cuda:
    torch.cuda.manual_seed(args.seed)

all_limg, all_rimg, all_ldisp, all_rdisp, test_limg, test_rimg, test_ldisp, test_rdisp = sf.sf_loader(args.data_path)

trainLoader = torch.utils.data.DataLoader(
    sf.myDataset(all_limg, all_rimg, all_ldisp, all_rdisp, training=True, color_transform=args.color_transform),
    batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)


fe_model = FE.VGG_Feature(fixed_param=True).eval()
adaptor = un.U_Net_v4(img_ch=256, output_ch=64).eval()
model = Agg.PSMAggregator(args.max_disp, udc=True).train()
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

if cuda:
    fe_model = nn.DataParallel(fe_model.cuda())
    adaptor = nn.DataParallel(adaptor.cuda())
    model = nn.DataParallel(model.cuda())

adaptor.load_state_dict(torch.load(args.load_path)['net'])
for p in adaptor.parameters():
    p.requires_grad = False

optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))


def train(imgL, imgR, gt_left, gt_right):
    imgL = torch.FloatTensor(imgL)
    imgR = torch.FloatTensor(imgR)
    gt_left = torch.FloatTensor(gt_left)
    gt_right = torch.FloatTensor(gt_right)

    if cuda:
        imgL, imgR, gt_left, gt_right = imgL.cuda(), imgR.cuda(), gt_left.cuda(), gt_right.cuda()

    optimizer.zero_grad()

    with torch.no_grad():
        left_fea = fe_model(imgL)
        right_fea = fe_model(imgR)

        left_fea = adaptor(left_fea)
        right_fea = adaptor(right_fea)

    loss1, loss2 = model(left_fea, right_fea, gt_left, training=True)
    loss1 = torch.mean(loss1)
    loss2 = torch.mean(loss2)
    loss = 0.1 * loss1 + loss2
    # loss = loss1

    loss.backward()
    optimizer.step()

    return loss1.item(), loss2.item()


def adjust_learning_rate(optimizer, epoch):
    if epoch <= 5:
        lr = 0.001
    else:
        lr = 0.0001
    # print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():

    # start_total_time = time.time()
    start_epoch = 1

    # checkpoint = torch.load('trained_ft_costAgg/checkpoint_1_v4.tar')
    # CostAggregator.load_state_dict(checkpoint['net'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch, args.epoch + start_epoch):
        print('This is %d-th epoch' % (epoch))
        total_train_loss1 = 0
        total_train_loss2 = 0
        adjust_learning_rate(optimizer, epoch)
        #

        for batch_id, (imgL, imgR, disp_L, disp_R) in enumerate(tqdm(trainLoader)):
            train_loss1, train_loss2 = train(imgL, imgR, disp_L, disp_R)
            total_train_loss1 += train_loss1
            total_train_loss2 += train_loss2
        avg_train_loss1 = total_train_loss1 / len(trainLoader)
        avg_train_loss2 = total_train_loss2 / len(trainLoader)
        print('Epoch %d average training loss1 = %.3f, average training loss2 = %.3f' %
              (epoch, avg_train_loss1, avg_train_loss2))

        state = {'fa_net': adaptor.state_dict(),
                 'net': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'epoch': epoch}
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        save_model_path = args.save_path + 'checkpoint_final_{}epoch.tar'.format(epoch)
        torch.save(state, save_model_path)

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

