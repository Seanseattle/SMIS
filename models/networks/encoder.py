"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
import torch

class ConvEncoder(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self, opt):
        super().__init__()

        kw = 3
        self.opt = opt
        pw = int(np.ceil((kw - 1.0) / 2))
        if opt.dataset_mode == 'cityscapes':
            ndf = 350
        elif opt.dataset_mode == 'ade20k':
            ndf = 151 * 4
        elif opt.dataset_mode == 'deepfashion':
            ndf = 256
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(self.opt.semantic_nc * 3, ndf, kw, stride=2, padding=pw, groups=self.opt.semantic_nc))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw, groups=self.opt.semantic_nc))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw, groups=self.opt.semantic_nc))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw, groups=self.opt.semantic_nc))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw, groups=self.opt.semantic_nc))
        if opt.crop_size >= 256:
            self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw, groups=self.opt.semantic_nc))
        self.so = s0 = 4
        self.fc_mu = nn.Conv2d(ndf * 8, 8 * self.opt.semantic_nc, stride=1, kernel_size=3, padding=1, groups=self.opt.semantic_nc)
        self.fc_var = nn.Conv2d(ndf * 8, 8 * self.opt.semantic_nc, stride=1, kernel_size=3, padding=1, groups=self.opt.semantic_nc)
        self.actvn = nn.LeakyReLU(0.2, False)


    def forward(self, x):
        bs = x.size(0)
        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        if self.opt.crop_size >= 256:
            x = self.layer6(self.actvn(x))
        x = self.actvn(x)

        # x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        return mu, logvar

class FcEncoder(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self, opt):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        if opt.crop_size >= 256:
            self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))

        self.so = s0 = 4
        self.fc_mu = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.fc_var = nn.Linear(ndf * 8 * s0 * s0, 256)

        self.actvn = nn.LeakyReLU(0.2, False)
        self.opt = opt

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        if self.opt.crop_size >= 256:
            x = self.layer6(self.actvn(x))
        x = self.actvn(x)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar