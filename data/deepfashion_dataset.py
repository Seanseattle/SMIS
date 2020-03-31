"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os.path
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset


class DeepfashionDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=8)
        opt, _ = parser.parse_known_args()
        return parser

    def get_paths(self, opt):
        root = opt.dataroot

        phase = 'test' if opt.phase == 'test' else 'train'

        label_dir = os.path.join(root, 'cihp_' + phase + '_mask')
        label_paths_all = make_dataset(label_dir, recursive=False)
        label_paths = [p for p in label_paths_all if p.endswith('.png')]

        image_dir = os.path.join(root, phase)
        image_paths = make_dataset(image_dir, recursive=False)
        instance_paths = []
        return label_paths, image_paths, instance_paths
