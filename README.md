# ICL++: Coarse-to-Fine Implicit Correspondence Learning for Image-to-Point Cloud Registration
This repo is the official implementation of extended version of our CVPR25 paper "Implicit Correspondence Learning for Image-to-Point Cloud Registration"
Xinjun Li, Wenfei Yang, Jiacheng Deng, Zhixin Cheng, Xu Zhou, Tianzhu Zhang

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
Since we made some modifications to the vision3d codebase — for example, the original vision3d does not support the nuScenes dataset — we provide the modified version used in our experiments.
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

## Data Preparation
You can download both the prepared KITTI and nuScenes datasets from the link provided by [CorrI2P](https://github.com/rsy6318/CorrI2P).

## Training
Our training process consists of two stages. In stage 1, we train only the GPDM (Geometric Prior-guided Overlapping Region Detection Module) using a classification loss and a frustum-pose loss for 20 epochs. In stage 2, we train the entire network for another 20 epochs, while keeping the parameters of the GPDM frozen.
### Training on KITTI
### Stage 1
The code is in `kitti/stage_1`. Use the following command for training.
```bash
CUDA_VISIBLE_DEVICES=0 python trainval.py
```

### Stage 2
The code is in `kitti/stage_2`. 
Save the checkpoint from Stage 1 to:
`kitti/stage_2/workspace/vision3d-output/stage_2/checkpoints/checkpoint.pth`

**Note: Make sure to save the checkpoint with the name `epoch-xx.pth` instead of `checkpoint.pth`, so that the training in Stage 2 can properly resume from the beginning.**

Use the following command for training.
```bash
CUDA_VISIBLE_DEVICES=0 python trainval.py --resume
```

### Training on nuScenes
### Stage 1
The code is in `nuscenes/stage_1`. Use the following command for training.
```bash
CUDA_VISIBLE_DEVICES=0 python trainval.py
```

### Stage 2
The code is in `nuscenes/stage_2`. 
Save the checkpoint from Stage 1 to:
`nuscenes/stage_2/workspace/vision3d-output/stage_2/checkpoints/checkpoint.pth`

**Note: Make sure to save the checkpoint with the name `epoch-xx.pth` instead of `checkpoint.pth`, so that the training in Stage 2 can properly resume from the beginning.**

Use the following command for training.
```bash
CUDA_VISIBLE_DEVICES=0 python trainval.py --resume
```

## Evaluation
### Stage 1
To evaluate the results of stage 1, you can run the following command:
```bash
bash eval.sh
```

### Stage 2
To evaluate the results of stage 2, you can run the following command:
```bash
bash eval.sh
```

## Acknowledgements
Our code is based on [2D3D-MATR](https://github.com/minhaolee/2D3DMATR), [vision3d](https://github.com/qinzheng93/vision3d) and [CorrI2P](https://github.com/rsy6318/CorrI2P). 
We thank the authors for their excellent work!

## Citation
This paper has been accepted to CVPR 2025.  
The official citation and link will be updated after the paper is published.