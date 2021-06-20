import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import numpy as np
from utils import align, wave2spec, spec2wave
from mir_eval.separation import bss_eval_sources
from models.mmdensenet import MMDenseNet
from models.unet import U_Net

def separate(model, mix, args):
    """
    - model: model to generate mask
    - mix: input mix (1, T)
    return: (drums, bass, other, vocals)
    """
    model.eval()
    spec, phase, tmplen = wave2spec(args, mix)

    freq, leng = spec.shape[0], spec.shape[1]
    tmp = np.zeros((freq, align(leng)), dtype=np.float32)
    spec = np.concatenate((spec, tmp), axis=1)

    spec = spec.transpose(1, 0).reshape(spec.shape[1]//128, 128, 512).transpose(0, 2, 1)

    spec_rst = []
    batchsize = 8
    for start_idx in range(0, spec.shape[0], batchsize):
        if start_idx + batchsize < spec.shape[0]:
            spec_i = torch.cuda.FloatTensor(spec[start_idx:start_idx+batchsize, None, :, :])
        else:
            spec_i = torch.cuda.FloatTensor(spec[start_idx:, None, :, :])

        masks = model(spec_i).detach().cpu()  # (1, 4, 512, T)
        spec_i = spec_i.cpu()

        if args.hard_mask:
            tmp = np.zeros(masks.shape, dtype="float32")
            tmp[masks > 0.5] = 1
            masks = torch.from_numpy(tmp)
        
        if args.extract:
            inst_specs = masks * spec_i
        else:
            inst_specs = (1.0 - masks) * spec_i
        
        specs = torch.cat([inst_specs[idx] for idx in range(inst_specs.shape[0])], axis=-1)
        spec_rst.append(specs)
    
    specs = torch.cat(spec_rst, dim=2)[..., :leng].numpy()
    out = [spec2wave(args, inst_spec, phase[:, :leng], tmplen) for inst_spec in specs]

    return out


def eval(ref, est):
    """ 
    ref: np.array (4, ...)
    est: np.array (4, ...)
    """
    sdr, sir, sar, _ = bss_eval_sources(ref, est)
    return sdr, sir, sar


class Model(object):
    def __init__(self, args, model_name):
        self.args = args
        if model_name == 'unet':
            self.model = U_Net().to(args.device)
        elif model_name == 'mmdensenet':
            self.model = MMDenseNet().to(args.device)
        self.log_model(self.model)

    def log_model(self, model):
        print(f"Model name: {self.model.__class__.__name__}")
        total_params = sum(p.numel() for p in model.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')


    def load_model(self, checkpoint):
        """Load the model checkpoints"""
        checkpoint = torch.load(checkpoint, map_location=self.args.device)
        self.model.load_state_dict(checkpoint['net'])

    def separate(self, mix, args, checkpoint):
        self.load_model(checkpoint)
        return separate(self.model, mix, args)

    def evaluate(self, mix, gt, checkpoint):
        """
        mix: test mixture (1, T)
        gt: test groundtruth (4, T)
        """
        self.load_model(checkpoint)
        insts = self.args.insts
        self.model.eval()
        predicts = separate(self.model, mix, self.args)
        ref, est = {}, {}
        for inst_idx, inst in enumerate(insts):
            ref[inst] = gt[inst_idx]
            est[inst] = predicts[inst_idx]
        ref = np.array([item for item in ref.values()])
        est = np.array([item for item in est.values()])
        sdr, sir, sar, _ = bss_eval_sources(ref, est)
        return sdr, sir, sar