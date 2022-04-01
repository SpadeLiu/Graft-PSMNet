import argparse
import torch
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import os
import copy
from tqdm import tqdm

from dataloader import sceneflow_loader as sf
import networks.submodule as sm
import networks.U_net as un
import networks.Aggregator as Agg
import networks.feature_extraction as FE
import loss_functions as lf


parser = argparse.ArgumentParser(description='GraftNet')
parser.add_argument('--no_cuda', action='store_true', default=False)
parser.add_argument('--gpu_id', type=str, default='0, 1')
parser.add_argument('--seed', type=str, default=0)
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--epoch', type=int, default=8)
parser.add_argument('--data_path', type=str, default='/media/data/dataset/SceneFlow/')
parser.add_argument('--save_path', type=str, default='trained_models/')
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


fe_model = sm.GwcFeature(out_c=64).train()
model = Agg.PSMAggregator(args.max_disp, udc=True).train()

if cuda:
    fe_model = nn.DataParallel(fe_model.cuda())
    model = nn.DataParallel(model.cuda())

params = [
    {'params': fe_model.parameters(), 'lr': 1e-3},
    {'params': model.parameters(), 'lr': 1e-3},
]
optimizer = optim.Adam(params, lr=1e-3, betas=(0.9, 0.999))


def train(imgL, imgR, gt_left, gt_right):
    imgL = torch.FloatTensor(imgL)
    imgR = torch.FloatTensor(imgR)
    gt_left = torch.FloatTensor(gt_left)
    gt_right = torch.FloatTensor(gt_right)

    if cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()
        gt_left, gt_right = gt_left.cuda(), gt_right.cuda()

    optimizer.zero_grad()

    left_fea = fe_model(imgL)
    right_fea = fe_model(imgR)

    loss1, loss2 = model(left_fea, right_fea, gt_left, training=True)

    loss1 = torch.mean(loss1)
    loss2 = torch.mean(loss2)

    loss = 0.1 * loss1 + loss2

    loss.backward()
    optimizer.step()

    return loss1.item(), loss2.item()


def adjust_learning_rate(optimizer, epoch):
    if epoch <= 10:
        lr = 0.001
    else:
        lr = 0.0001
    # print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():

    # start_total_time = time.time()
    start_epoch = 1

    # checkpoint = torch.load('trained_gwcAgg/checkpoint_5_v1.tar')
    # model.load_state_dict(checkpoint['net'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # start_epoch = checkpoint['epoch'] + 1
    # new_dict = {}
    # for k, v in checkpoint['fe_net'].items():
    #     k = "module." + k
    #     new_dict[k] = v
    # fe_model.load_state_dict(new_dict)
    # optimizer_fe.load_state_dict(checkpoint['fe_optimizer'])

    for epoch in range(start_epoch, args.epoch + start_epoch):
        print('This is %d-th epoch' % (epoch))
        total_train_loss1 = 0
        total_train_loss2 = 0
        adjust_learning_rate(optimizer, epoch)

        for batch_id, (imgL, imgR, disp_L, disp_R) in enumerate(tqdm(trainLoader)):
            train_loss1, train_loss2 = train(imgL, imgR, disp_L, disp_R)
            total_train_loss1 += train_loss1
            total_train_loss2 += train_loss2
        avg_train_loss1 = total_train_loss1 / len(trainLoader)
        avg_train_loss2 = total_train_loss2 / len(trainLoader)
        print('Epoch %d average training loss1 = %.3f, average training loss2 = %.3f' %
              (epoch, avg_train_loss1, avg_train_loss2))

        state = {'net': model.state_dict(),
                 'fe_net': fe_model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'epoch': epoch}
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        save_model_path = args.save_path + 'checkpoint_baseline_{}epoch.tar'.format(epoch)
        torch.save(state, save_model_path)

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

