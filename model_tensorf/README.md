# TensoRF
### Reference: [TensoRF: Tensorial Radiance Fields](https://github.com/apchenstu/TensoRF) 


## Installation

#### Tested on Ubuntu 20.04 + Pytorch 1.10.1 

Install environment:
```
conda create -n TensoRF python=3.8
conda activate TensoRF
pip install torch torchvision
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard
```


## Dataset
* [Synthetic-NeRF](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) 
* [Synthetic-NSVF](https://dl.fbaipublicfiles.com/nsvf/dataset/Synthetic_NSVF.zip)
* [Tanks&Temples](https://dl.fbaipublicfiles.com/nsvf/dataset/TanksAndTemple.zip)
* [Forward-facing](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)



## How to use
### Training

```
python run.py --config configs/lego.txt
```

### Rendering

```
python train.py --config configs/lego.txt --ckpt path/to/your/checkpoint --render_only 1 --render_test 1 
```

## Pipeline
* Preprocessing data 
    * rays, rgbs 리턴
* Volume Rendering 
    1. ray_o, ray_d를 input으로 하는`TensorBase.sample_ray` function을 사용해서 ray를 sampling
    2. sampling된 ray를 model의 input으로 사용하여 rgb_map, depth_map 리턴 (사용 가능한 model: `TensorVM`, `TensorVMSplit`, `TensorCP`)
    3. 특정 step에서 upsampling 진행 
* Loss
    1. rgb_map과 rgb_train 간의 L2 loss
    2. regularization term: L1 norm loss, TV loss

