Semantically Multi-modal Image Synthesis
---
### [Project page](http://seanseattle.github.io/SMIS) / [Paper](https://arxiv.org/abs/2003.12697)  / [Demo](https://www.youtube.com/watch?v=uarUonGi_ZU&t=2s)
![gif demo](docs/imgs/smis.gif) \
Semantically Multi-modal Image Synthesis(CVPR2020). \
Zhen Zhu, Zhiliang Xu, Ansheng You, Xiang Bai

### Requirements
---
- torch>=1.0.0
- torchvision
- dominate
- dill
- scikit-image
- tqdm
- opencv-python

### Getting Started
----
#### Data Preperation
**DeepFashion** \
**Note:** We provide an example of the [DeepFashion](https://drive.google.com/open?id=1ckx35-mlMv57yzv47bmOCrWTm5l2X-zD) dataset. That is slightly different from the DeepFashion used in our paper due to the impact of the COVID-19.


**Cityscapes** \
The Cityscapes dataset can be downloaded at [here](https://www.cityscapes-dataset.com/)

**ADE20K** \
The ADE20K dataset can be downloaded at [here](http://sceneparsing.csail.mit.edu/) 

#### Test/Train the models
Download the tar of the pretrained models from the [Google Drive Folder](https://drive.google.com/open?id=1og_9By_xdtnEd9-xawAj4jYbXR6A9deG). Save it in `checkpoints/` and unzip it.
There are deepfashion.sh, cityscapes.sh and ade20k.sh in the scripts folder. Change the parameters like `--dataroot` and so on, then comment or uncomment some code to test/train model. 
And you can specify the `--test_mask` for SMIS test.

  
### Acknowledgments
---
Our code is based on the popular [SPADE](https://github.com/NVlabs/SPADE)
