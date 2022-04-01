import torch
import torch.nn as nn
import math


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=32)
        self.Conv2 = conv_block(ch_in=32, ch_out=64)
        self.Conv3 = conv_block(ch_in=64, ch_out=128)
        self.Conv4 = conv_block(ch_in=128, ch_out=256)
        # self.Conv5 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=256, ch_out=256)

        # self.Up5 = up_conv(ch_in=512, ch_out=256)
        self.Up5 = up_conv(ch_in=256, ch_out=256)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)

        self.Up3 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)

        self.Up2 = up_conv(ch_in=64, ch_out=32)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32, output_ch, kernel_size=1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class U_Net_v2(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net_v2, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=32)
        self.Conv2 = conv_block(ch_in=32, ch_out=64)
        self.Conv3 = conv_block(ch_in=64, ch_out=128)
        self.Conv4 = conv_block(ch_in=128, ch_out=256)

        self.Up4 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)

        self.Up3 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)

        self.Up2 = up_conv(ch_in=64, ch_out=32)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32, output_ch, kernel_size=1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        d4 = self.Up4(x4)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class U_Net_v3(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net_v3, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv0 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv1 = conv_block(ch_in=64, ch_out=128)
        self.Conv2 = conv_block(ch_in=128, ch_out=256)

        self.Up5 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv5 = conv_block(ch_in=256, ch_out=128)

        self.Up4 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv4 = conv_block(ch_in=128, ch_out=64)

        self.Up3 = up_conv(ch_in=64, ch_out=32)
        self.Up_conv3 = conv_block(ch_in=32, ch_out=32)

        self.Up2 = up_conv(ch_in=32, ch_out=32)
        self.Up_conv2 = conv_block(ch_in=32, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32, output_ch, kernel_size=1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        x0 = self.Conv0(x)                  # 64 channels

        x1 = self.Conv1(x0)                 # 128 channels
        x1 = self.Maxpool(x1)               # 1/8 resolution

        x2 = self.Conv2(x1)                 # 256 channels
        x2 = self.Maxpool(x2)               # 1/16 resolution

        d4 = self.Up5(x2)                   # 1/8 resolution
        d4 = torch.cat((x1, d4), dim=1)
        d4 = self.Up_conv5(d4)              # 128 channels

        d3 = self.Up4(d4)                   # 1/4 resolution
        d3 = torch.cat((x0, d3), dim=1)
        d3 = self.Up_conv4(d3)              # 64 channels

        d2 = self.Up3(d3)                   # 1/2 resolution
        d2 = self.Up_conv3(d2)              # 32 channels

        d1 = self.Up2(d2)                   # 1/2 resolution
        d1 = self.Up_conv2(d1)              # 32 channels

        d0 = self.Conv_1x1(d1)

        return d0


class U_Net_v4(nn.Module):
    def __init__(self, img_ch, output_ch):
        super(U_Net_v4, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=32)
        self.Conv2 = conv_block(ch_in=32, ch_out=64)
        self.Conv3 = conv_block(ch_in=64, ch_out=128)

        self.Conv4 = conv_block(ch_in=128, ch_out=128)

        self.Up4 = conv_block(ch_in=128, ch_out=128)
        self.Up_conv4 = up_conv(ch_in=256, ch_out=64)

        self.Up3 = conv_block(ch_in=64, ch_out=64)
        self.Up_conv3 = up_conv(ch_in=128, ch_out=32)

        self.last_conv = nn.Conv2d(64, output_ch, 1, 1, 0, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # nn.init.kaiming_normal_(m, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        x1 = self.Conv1(x)              # 32, 1/4

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)             # 64, 1/8

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)             # 128, 1/16

        x4 = self.Conv4(x3)             # 128, 1/16

        d4 = self.Up4(x4)               # 128, 1/16
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)          # 64, 1/8

        d3 = self.Up3(d4)               # 64, 1/8
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)          # 32, 1/4

        d2 = torch.cat((x1, d3), dim=1)
        d2 = self.last_conv(d2)

        return d2


class LinearProj(nn.Module):
    def __init__(self, in_c, out_c):
        super(LinearProj, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, 1, 0, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 1, 1, 0, 1))
        # self.conv = nn.Conv2d(in_c, out_c, 1, 1, 0, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


if __name__ == '__main__':
    a = torch.rand(2, 3, 64, 128).cuda()
    net = U_Net_v3(img_ch=3, output_ch=4).cuda()
    b = net(a)
    print(b.shape)