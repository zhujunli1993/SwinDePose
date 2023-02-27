# SwinDePose. 

## Table of Contents  

- [SwinDePose](#swindepose)
  - [Table of Content](#table-of-content)
  - [Introduction & Citation](#introduction--citation)
  - [Installation](#installation)
  - [Code Structure](#code-structure)
  - [Datasets](#datasets)
  - [Training and evaluating](#training-and-evaluating)
    - [Training on the LineMOD Dataset](#training-on-the-linemod-dataset)
    - [Evaluating on the LineMOD Dataset](#evaluating-on-the-linemod-dataset)
    - [Demo/visualizaion on the LineMOD Dataset](#demovisualizaion-on-the-linemod-dataset)
    - [Training on the Occ-LineMod Dataset](#training-on-the-occ_linemod-dataset)
    - [Evaluating on the Occ-LineMod Dataset](#evaluating-on-the-Occ-LineMod-dataset)
    - [Demo/visualization on the Occ-LineMod Dataset](#demovisualization-on-the-Occ-LineMod-dataset)
  - [Results](#results)
  - [License](#license)

This is the official source code for the SwinDePose: Depth-based Object 6DoF Pose Estimation using Swin Transformers.
To preview the MD file, press ctrl+k+v

## Introduction & Citation
<div align=center><img width="100%" src="figs/overview_one.PNG"/></div>
### Run our FFB6D docker 
```bash 
sudo nvidia-docker run --gpus all --ipc=host --shm-size 50G --ulimit memlock=-1 --name swin-ffb -it --rm -v /raid/home/zl755:/workspace zhujunli/ffb6d:latest
```
or 
```bash 
sudo docker exec -it swin-ffb /bin/bash
```



### Generate LindMod Normal Angles Images 
To generate synthetic normal angles images: 
1. Open raster_triangle folder.
2. Link the Linemod to the current folder. 
```bash 
ln -s path_to_Linemod_preprocessed ./Linemod_preprocessed
```
Don't have to do it every time. 
3. Render renders_nrm/ data. For example, for phone class.
```bash 
python3 rgbd_renderer.py --cls phone --render_num 10000
```
4. Render fuse_nrm/ data. For example, for phone class.
```bash 
python3 fuse.py --cls phone --fuse_num 10000
```
To generate real/ normal angles images: Run the codes within the folder: datasets/linemod
```bash 
python -m create_angle_npy.py --cls_num your_cls_num --train_list 'train.txt' --test_list 'test.txt'
```

### Train and inference the model on YCBV dataset
0. Download codes from git.
1. Create and go into the docker container.
2. Go into the ffb6d folder, e.g.:
```bash 
cd /workspace/REPO/pose_estimation/ffb6d
```
3. To train the model, run: 
```bash 
sh scripts/train_ycb.sh
```
4. To infer the model, run:
```bash 
sh scripts/test_ycb.sh
```