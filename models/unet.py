import torch.nn as nn
import torch.nn.functional as F
import torch


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 5, padding=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 5, padding=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.dropout = dropout
    def forward(self, input):
        out = self.conv(input)
        if self.dropout:
            out = nn.Dropout(0.4)(out)
        return out



class U_Net(nn.Module):
    def __init__(self,out_channels=5):
        super(U_Net, self).__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()

        self.conv0 = Conv(1, nb_filter[0])
        self.conv1 = Conv(nb_filter[0], nb_filter[1])
        self.conv2 = Conv(nb_filter[1], nb_filter[2])
        self.conv3 = Conv(nb_filter[2], nb_filter[3])
        self.conv4 = Conv(nb_filter[3], nb_filter[4])

        self.deconv0 = Conv(nb_filter[4]*2, nb_filter[3], True)
        self.deconv1 = Conv(nb_filter[3]*2, nb_filter[2], True)
        self.deconv2 = Conv(nb_filter[2]*2, nb_filter[1], True)
        self.deconv3 = Conv(nb_filter[1]*2, nb_filter[0], False)
        self.deconv4 = Conv(nb_filter[0]*2, nb_filter[0], False)
        self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)



    def forward(self, input):
        x0 = self.conv0(input)   # (n, 32, 512, 256)
        x0_m = self.pool(x0)    # (n, 32, 256, 128)
        x1 = self.conv1(x0_m)   # (n, 64, 256, 128)
        x1_m = self.pool(x1)    # (n, 64, 128, 64)
        x2 = self.conv2(x1_m)   # (n, 128, 128, 64)
        x2_m = self.pool(x2)    # (n, 128, 64, 32)
        x3 = self.conv3(x2_m)   # (n, 256, 64, 32)
        x3_m = self.pool(x3)    # (n, 256, 32, 16)
        x4 = self.conv4(x3_m)   # (n, 512, 32, 16)
        x4_m = self.pool(x4)    # (n, 512, 16, 8)

        h = self.deconv0(torch.cat((x4, self.up(x4_m)), 1))   # (n, 256, 32, 16)
        h = self.deconv1(torch.cat((x3, self.up(h)), 1))      # (n, 128, 64, 32)
        h = self.deconv2(torch.cat((x2, self.up(h)), 1))
        h = self.deconv3(torch.cat((x1, self.up(h)), 1))
        h = self.deconv4(torch.cat((x0, self.up(h)), 1))

        h = self.final(h)
        h = self.sigmoid(h)

        return h
