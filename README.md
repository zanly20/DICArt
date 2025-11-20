# DICArt: Advancing Category-level Articulated Object Pose Estimation in Discrete State-Spaces

## Overview
![alt text](assets/pipeline.png)

<div align="center">
<h3>
<a href="https://zanly20.github.io">Li Zhang</a>,  Mingyu Mei, Ailing Zeng, Xianhui Meng, Yan Zhong, Xinyuan Song, Liu Liu, Rujing Wang, Zaixin He, Cewu Lu
<br>
<br>
<a href='https://sites.google.com/view/dicart'>
  <img src='https://img.shields.io/badge/Project-Page-green?style=flat&logo=googlechrome&logoColor=green' alt='Project Page'>
</a>
<br>
</h3>
</div>

## Requirements

- Ubuntu 22.04
- Python 3.9
- CUDA 11.8
- NVIDIA RTX 3090

## Installation

- ### Install pytorch
Create a new conda environment and activate the environment.
```bash
conda create -n DICArt python=3.9
conda activate DICArt
```

``` bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```
- ### Install from requirements.txt
``` bash
pip install -r requirements.txt 
```

- ### Install pytorch3d from a local clone
``` bash
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e .
```

- ### Compile pointnet2
``` bash
cd networks/pts_encoder/pointnet2_utils/pointnet2
python setup.py install
```

## Training
Set the parameter '--data_path' in scripts/train.sh 

- ### Training network

``` bash
bash scripts/train.sh
```
- ### Eval network
``` bash
bash scripts/eval.sh
```
