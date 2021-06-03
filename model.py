import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import numpy as np
from utils import align, wave2spec, spec2wave
from mir_eval.separation import bss_eval_sources

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


def separate(generator, mix, args):
    """
    - generator: model to generate mask
    - mix: input mix (1, T)
    return: (drums, bass, )
    """
    # mix audio ---> mix spec
    # mix spec ---> instrument spec
    # instrument spec ---> instrument audio
    # generator = generator.to('cpu')
    generator.eval()
    spec, phase, tmplen = wave2spec(args, mix)      # mix spec (1, 1, F, T)

    freq, leng = spec.shape[0], spec.shape[1]
    tmp = np.zeros((freq, align(leng)), dtype=np.float32)
    spec = np.concatenate((spec, tmp), axis=1)
    spec = torch.FloatTensor(spec[None, None, :, :])
    
    generator, spec = generator.cpu(), spec.cpu()
    masks = generator(spec)  # (1, 4, 512, T)
    
    if args.hard_mask:
        tmp = np.zeros(masks.shape, dtype="float32")
        tmp[masks > 0.5] = 1
        masks = torch.from_numpy(tmp)
    
    if args.extract:
        inst_specs = masks * spec
    else:
        inst_specs = (1.0 - masks) * spec
    
    specs = inst_specs[0, :, :, :leng].detach().cpu().numpy()   # (4, 512, T)

    inst_specs = tuple([specs[index, ...] for index in range(len(args.insts))])
    out = []
    for inst_spec in inst_specs:
        out.append(spec2wave(args, inst_spec, phase[:, :leng], tmplen))
    return out


def eval(ref, est):
    """ 
    ref: np.array (4, ...)
    est: np.array (4, ...)
    """
    sdr, sir, sar, _ = bss_eval_sources(ref, est)
    return sdr, sir, sar


class Model(object):
    def __init__(self, args):
        self.args = args
        self.unet = U_Net().to(args.device)
        self.log_model(self.unet)

    def log_model(self, model):
        print(f"Model name: {self.unet.__class__.__name__}")
        total_params = sum(p.numel() for p in model.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')


    def load_model(self, checkpoint):
        """Load the generator checkpoints"""
        checkpoint = torch.load(checkpoint, map_location=self.args.device)
        self.unet.load_state_dict(checkpoint['net'])

    def separate(self, mix, args, checkpoint):
        self.load_model(checkpoint)
        return separate(self.unet, mix, args)

    def evaluate(self, mix, gt, checkpoint):
        """
        mix: test mixture (1, T)
        gt: test groundtruth (4, T)
        """
        self.load_model(checkpoint)
        insts = self.args.insts
        self.unet.eval()
        predicts = separate(self.unet, mix, self.args)
        ref, est = {}, {}
        for inst_idx, inst in enumerate(insts):
            ref[inst] = gt[inst_idx]
            est[inst] = predicts[inst_idx]
        ref = np.array([item for item in ref.values()])
        est = np.array([item for item in est.values()])
        sdr, sir, sar, _ = bss_eval_sources(ref, est)
        return sdr, sir, sar
    
    def spec_plot(self, mix, gt, iteration):

        insts = self.args.insts
        self.unet.eval()
        spec, phase, tmplen = wave2spec(self.args, mix)      # mix spec (1, 1, F, T)
        freq, leng = spec.shape[0], spec.shape[1]
        tmp = np.zeros((freq, align(leng)), dtype=np.float32)
        spec = np.concatenate((spec, tmp), axis=1)
        spec = torch.cuda.FloatTensor(spec[None, None, :, :])
        inst_specs = self.unet(spec) * spec  # (1, 4, 512, T)
        spec_estimate = inst_specs[0, :, :, :leng].detach().cpu().numpy()   # (4, 512, T)

        for inst_idx, inst in enumerate(insts):
            spec_gt = wave2spec(self.args, gt[inst_idx])[0]
            self.logger.log_img(spec_gt, spec_estimate[inst_idx], iteration, inst)