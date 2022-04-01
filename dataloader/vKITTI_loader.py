import torch.utils.data as data
import torchvision.transforms as transforms
import os
from PIL import Image
import random
import numpy as np


def vkt_loader(filepath):
    all_limg = []
    all_rimg = []
    all_disp = []

    img_path = os.path.join(filepath, 'vkitti_2.0.3_rgb')
    depth_path = os.path.join(filepath, 'vkitti_2.0.3_depth')

    for scene in os.listdir(img_path):
        img_scenes_path = os.path.join(img_path, scene, 'clone/frames/rgb')
        depth_scenes_path = os.path.join(depth_path, scene, 'clone/frames/depth')

        for name in os.listdir(os.path.join(img_scenes_path, 'Camera_0')):
            all_limg.append(os.path.join(img_scenes_path, 'Camera_0', name))
            all_rimg.append(os.path.join(img_scenes_path, 'Camera_1', name))
            all_disp.append(os.path.join(depth_scenes_path, 'Camera_0',
                                         name.replace('jpg', 'png').replace('rgb', 'depth')))

    total_num = len(all_limg)
    train_length = int(total_num * 0.75)

    train_limg = all_limg[:train_length]
    train_rimg = all_rimg[:train_length]
    train_disp = all_disp[:train_length]

    val_limg = all_limg[train_length:]
    val_rimg = all_rimg[train_length:]
    val_disp = all_disp[train_length:]

    return train_limg, train_rimg, train_disp, val_limg, val_rimg, val_disp


def img_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    return Image.open(path)


class vkDataset(data.Dataset):

    def __init__(self, left, right, left_disp, training, imgloader=img_loader, disploader=disparity_loader):
        self.left = left
        self.right = right
        self.left_disp = left_disp
        self.imgloader = imgloader
        self.disploader = disploader
        self.training = training
        self.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        left_disp = self.left_disp[index]

        limg = self.imgloader(left)
        rimg = self.imgloader(right)
        ldisp = self.disploader(left_disp)

        if self.training:
            w, h = limg.size
            tw, th = 512, 256

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            limg = limg.crop((x1, y1, x1 + tw, y1 + th))
            rimg = rimg.crop((x1, y1, x1 + tw, y1 + th))

            limg = self.transform(limg)
            rimg = self.transform(rimg)

            baseline, fx, fy = 0.532725, 725.0087, 725.0087
            camera_params = {'baseline': baseline,
                             'fx': fx,
                             'fy': fy}

            ldepth = np.ascontiguousarray(ldisp, dtype=np.float32) / 100.
            ldisp = baseline * fy / ldepth
            ldisp = ldisp[y1:y1 + th, x1:x1 + tw]

            return limg, rimg, ldisp, ldisp

        else:
            w, h = limg.size

            limg = limg.crop((w-1232, h-368, w, h))
            rimg = rimg.crop((w-1232, h-368, w, h))
            ldisp = ldisp.crop((w-1232, h-368, w, h))

            limg = self.transform(limg)
            rimg = self.transform(rimg)

            baseline, fx, fy = 0.532725, 725.0087, 725.0087
            ldepth = np.ascontiguousarray(ldisp, dtype=np.float32) / 100.
            ldisp = baseline * fy / ldepth

            return limg, rimg, ldisp, ldisp

    def __len__(self):
        return len(self.left)


if __name__ == '__main__':

    path = '/media/data2/Dataset/vKITTI2/'
    a, b, c, d, e, f = vkt_loader(path)
    print(len(a))