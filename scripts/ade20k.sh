#python train.py --name ade20k_smis --dataset_mode ade20k --dataroot /home/zlxu/data/ADEChallengeData2016 --no_instance  \
#--gpu_ids 0,1,2,3 --ngf 64 --batchSize 4 --use_vae --niter 100 --niter_decay 100 --model smis --netE conv --netG ADE20K
#
python test.py --name ade20k_smis --dataset_mode ade20k --dataroot /home/zlxu/data/ADEChallengeData2016 --no_instance  \
--gpu_ids 0 --ngf 64 --batchSize 2 --model smis --netG ADE20K