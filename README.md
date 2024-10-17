# Sparse-to-dense Multimodal Image Registration via Multi-Task Learning
This repository contains the code for the ICML'24 paper "Sparse-to-dense Multimodal Image Registration via Multi-Task Learning". [(paper link)](https://openreview.net/pdf?id=q0vILV7zAw)

# Setup
1. torch
```
conda create --name SDME python==3.9.18 && \
conda activate SDME && \
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```
2. other requirements
```
pip install -r requirements.txt
```

# Datasets
1. Create your own DATASET_ROOT_PATH.
2. Put each dataset under DATASET_ROOT_PATH. The final file structure is like:
```
- DATASET_ROOT_PATH/
 - MSCOCO/
 - GoogleEarth/
 - VIS_NIR/
 - VIS_IR_drone/
```

**Download link:**

MSCOCO, GoogleEarth: download as the instructions of [DeepLK](https://github.com/placeforyiming/CVPR21-Deep-Lucas-Kanade-Homography)

[VIS_NIR](https://drive.google.com/file/d/1h7aTwmQL9xrVgWjEhdSIgmJAT6oJy4NB/view?usp=drive_link)

[VIS_IR_drone](https://drive.google.com/file/d/1h7aTwmQL9xrVgWjEhdSIgmJAT6oJy4NB/view?usp=drive_link)



# Usage
We provide the training and testing scripts for each dataset individually.

**Note:**
 - Replace [DATASET] with one of the following choices: [MSCOCO, GoogleEarth, VIS_NIR, VIS_IR].
 - Revise the DATASET_ROOT_PATH in the script to your own data path. 

1. train
```
cd scripts/train/ && \
bash train_[DATASET].sh 
```
2. test (pre-trained checkpoints are provided in checkpoints/)
```
cd scripts/test/ && \
bash test_[DATASET].sh 
```

# Citation
```
@inproceedings{zhangsparse,
  title={Sparse-to-dense Multimodal Image Registration via Multi-Task Learning},
  author={Zhang, Kaining and Ma, Jiayi},
  booktitle={Proceedings of the International Conference on Machine Learning},
  pages={59490--59504},
  year={2024}
}
```

# Acknowledgement
The codes are based on [R2D2](https://github.com/naver/r2d2) and [MultiObjectiveOptimization](https://github.com/isl-org/MultiObjectiveOptimization), we thank the authors for the contribution.
