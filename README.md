# ICL++: Coarse-to-Fine Implicit Correspondence Learning for Image-to-Point Cloud Registration
This repo is the official implementation of extended version of our CVPR25 paper "Implicit Correspondence Learning for Image-to-Point Cloud Registration"

The extended version primarily includes the following additions:
1. We design a coarse-to-fine strategy to refine the image-to-point cloud correspondence and camera pose, which can improve the performance with smaller computational cost.
2. We conduct more experiments to clarify the effectiveness and limitation of the proposed method.

We will soon release a preprint about the extended paper where you can find more details.

## Installation
Please use the following command for installation.

```bash
# It is recommended to create a new environment
conda create -n ICLI2P python==3.8
conda activate ICLI2P

# 2. Install vision3d following https://github.com/qinzheng93/vision3d
```
The code has been tested on Python 3.8, PyTorch 1.13.1, Ubuntu 22.04, GCC 11.3 and CUDA 11.7, but it should work with other configurations.

## Pre-trained Weights
We provide pre-trained weights from [BaiduYun](https://pan.baidu.com/s/16BVtBUjiBTNy-UdbrIHYng?pwd=54s4)(extraction code: 54s4). Please download the latest weights and place them into the appropriate directory:
```
kitti/ (or nuscenes/)
└── stage_1/ (or stage_2/)
    └── workspace/
        └── vision3d-output/
            └── stage_1/ (or stage_2/)
                └── checkpoints/
```
Make sure to choose the correct dataset (kitti or nuscenes) and stage (stage_1 or stage_2) accordingly.