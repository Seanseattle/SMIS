#!/usr/bin/env bash
#python train.py --name cityscapes_smis --dataset_mode cityscapes --dataroot /home/zlxu/data/cityscapes  \
#--gpu_ids 0,1,2,3 --ngf 280 --batchSize 4 --niter 100 --niter_decay 100 --netG Cityscapes --model smis --netE conv --use_vae

python test.py --name cityscapes_smis --dataset_mode cityscapes --dataroot /home/zlxu/data/cityscapes  \
--gpu_ids 1 --ngf 280 --batchSize 4 --netG Cityscapes --model smis