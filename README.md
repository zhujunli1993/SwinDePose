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
[SwinDePose] is a general framework for representation learning from a depth image, and we applied it to the 6D pose estimation task by cascading downstream prediction headers for instance semantic segmentation and 3D keypoint voting prediction from FFB6D.

Before the representation learning stage of SwinDePose, we build **normal vector angles image generation** module to generate normal vector angles images from depth images. Besides, depth images are lifted to point clouds by camera intrinsic parameters K. Then, the normal vector angles images and point clouds are fed into images and point clouds feature extraction networks to learn representations. Moreover, the learned embeddings from normal vector angles images and point clouds are fed into 3D keypoints localization module and instance segmentation module. Finally, a least-squares fitting manner is applied to estimate 6D poses. 

## Installation - From docker 
- Pull docker image from docker hub
```bash 
docker pull zhujunli/swin-pose
```
- Run our swin-pose docker
```bash 
sudo nvidia-docker run --gpus all --ipc=host --shm-size 50G --ulimit memlock=-1 --name swin-ffb -it --rm -v your_workspace_directory:/workspace zhujunli/swin-pose
```

## Code Structure

<details>
  <summary>[Click to expand]</summary>

- **ffb6d**
  - **ffb6d/common.py**: Common configuration of dataset and models, eg. dataset path, keypoints path, batch size and so on.
  - **ffb6d/datasets**
    - **ffb6d/datasets/linemod/**
      - **ffb6d/datasets/linemod/linemod_dataset.py**: Data loader for LineMOD dataset.
      - **ffb6d/datasets/linemod/dataset_config/models_info.yml**: Object model info of LineMOD dataset.
      - **ffb6d/datasets/linemod/kps_orb9_fps**
        - **ffb6d/datasets/linemod/kps_orb9_fps/{obj_name}_8_kps.txt**: ORB-FPS 3D keypoints of an object in the object coordinate system.
        - **ffb6d/datasets/linemod/kps_orb9_fps/{obj_name}_corners.txt**: 8 corners of the 3D bounding box of an object in the object coordinate system.
    - **ffb6d/datasets/ycb**
      - **ffb6d/datasets/ycb/ycb_dataset.py**： Data loader for YCB_Video dataset.
        - **ffb6d/datasets/ycb/dataset_config/classes.txt**: Object list of YCB_Video dataset.
        - **ffb6d/datasets/ycb/dataset_config/radius.txt**: Radius of each object in YCB_Video dataset.
        - **ffb6d/datasets/ycb/dataset_config/train_data_list.txt**: Training set of YCB_Video datset.
        - **ffb6d/datasets/ycb/dataset_config/test_data_list.txt**: Testing set of YCB_Video dataset.
      - **ffb6d/datasets/ycb/ycb_kps**
        - **ffb6d/datasets/ycb/ycb_kps/{obj_name}_8_kps.txt**: ORB-FPS 3D keypoints of an object in the object coordinate system.
        - **ffb6d/datasets/ycb/ycb_kps/{obj_name}_corners.txt**: 8 corners of the 3D bounding box of an object in the object coordinate system.
  - **ffb6d/models**
    - **ffb6d/models/ffb6d.py**: Network architecture of the proposed FFB6D.
    - **ffb6d/models/cnn**
      - **ffb6d/models/cnn/extractors.py**: Resnet backbones.
      - **ffb6d/models/cnn/pspnet.py**: PSPNet decoder.
      - **ffb6d/models/cnn/ResNet_pretrained_mdl**: Resnet pretraiend model weights.
    - **ffb6d/models/loss.py**: loss calculation for training of FFB6D model.
    - **ffb6d/models/pytorch_utils.py**: pytorch basic network modules.
    - **ffb6d/models/RandLA/**: pytorch version of RandLA-Net from [RandLA-Net-pytorch](https://github.com/qiqihaer/RandLA-Net-pytorch)
  - **ffb6d/utils**
    - **ffb6d/utils/basic_utils.py**: basic functions for data processing, visualization and so on.
    - **ffb6d/utils/meanshift_pytorch.py**: pytorch version of meanshift algorithm for 3D center point and keypoints voting. 
    - **ffb6d/utils/pvn3d_eval_utils_kpls.py**: Object pose esitimation from predicted center/keypoints offset and evaluation metrics.
    - **ffb6d/utils/ip_basic**: Image Processing for Basic Depth Completion from [ip_basic](https://github.com/kujason/ip_basic).
    - **ffb6d/utils/dataset_tools**
      - **ffb6d/utils/dataset_tools/DSTOOL_README.md**: README for dataset tools.
      - **ffb6d/utils/dataset_tools/requirement.txt**: Python3 requirement for dataset tools.
      - **ffb6d/utils/dataset_tools/gen_obj_info.py**: Generate object info, including SIFT-FPS 3d keypoints, radius etc.
      - **ffb6d/utils/dataset_tools/rgbd_rnder_sift_kp3ds.py**: Render rgbd images from mesh and extract textured 3d keypoints (SIFT/ORB).
      - **ffb6d/utils/dataset_tools/utils.py**: Basic utils for mesh, pose, image and system processing.
      - **ffb6d/utils/dataset_tools/fps**: Furthest point sampling algorithm.
      - **ffb6d/utils/dataset_tools/example_mesh**: Example mesh models.
  - **ffb6d/train_ycb.py**: Training & Evaluating code of FFB6D models for the YCB_Video dataset.
  - **ffb6d/demo.py**: Demo code for visualization.
  - **ffb6d/train_ycb.sh**: Bash scripts to start the training on the YCB_Video dataset.
  - **ffb6d/test_ycb.sh**: Bash scripts to start the testing on the YCB_Video dataset.
  - **ffb6d/demo_ycb.sh**: Bash scripts to start the demo on the YCB_Video_dataset.
  - **ffb6d/train_lm.py**: Training & Evaluating code of FFB6D models for the LineMOD dataset.
  - **ffb6d/train_lm.sh**: Bash scripts to start the training on the LineMOD dataset.
  - **ffb6d/test_lm.sh**: Bash scripts to start the testing on the LineMOD dataset.
  - **ffb6d/demo_lm.sh**: Bash scripts to start the demo on the LineMOD dataset.
  - **ffb6d/train_log**
    - **ffb6d/train_log/ycb**
      - **ffb6d/train_log/ycb/checkpoints/**: Storing trained checkpoints on the YCB_Video dataset.
      - **ffb6d/train_log/ycb/eval_results/**: Storing evaluated results on the YCB_Video_dataset.
      - **ffb6d/train_log/ycb/train_info/**: Training log on the YCB_Video_dataset.
- **requirement.txt**: python3 environment requirements for pip3 install.
- **figs/**: Images shown in README.

</details>

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