"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2pixModel
from models.smis_model import SmisModel
from util.visualizer import Visualizer
from util import html
from tqdm import tqdm

opt = TestOptions().parse()
# print(opt)
dataloader = data.create_dataloader(opt)
if opt.model == 'smis':
    model = SmisModel(opt)
elif opt.model == 'pix2pix':
    model = Pix2pixModel(opt)
model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))
for i, data_i in tqdm(enumerate(dataloader)):
    generated = model(data_i, mode='inference')
    img_path = data_i['path']
    for b in range(generated[0].shape[0]):
        if opt.test_times == 1:
            visuals = OrderedDict([('synthesized_image', generated[0][b])])
        else:
            visuals = OrderedDict([('input_label', data_i['label'][b]),
                                   ('real_image', data_i['image'][b]),
                                   ])
            for t in range(len(generated)):
                visuals['synthesized_image_' + str(t)] = generated[t][b]
        visualizer.save_images(webpage, visuals, img_path[b:b + 1])
webpage.save()
