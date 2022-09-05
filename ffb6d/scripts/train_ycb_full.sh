#!/bin/bash
# checkpoint to resume.
# tst_mdl=train_log/ycb_all_real_newtrainlist_onlydata/checkpoints/FFB6D_best.pth.tar  
GPU_NUM=0
GPU_COUNT=1
NAME='vtesting'
WANDB_PROJ='pose_estimation'
export CUDA_VISIBLE_DEVICES=$GPU_NUM
python -m torch.distributed.launch --nproc_per_node=$GPU_COUNT apps/train_ycb_full.py \
    --gpus=$GPU_COUNT \
    --wandb_proj $WANDB_PROJ \
    --wandb_name $NAME \
    --num_threads 0 \
    --data_root '/workspace/DATA/YCB_Dataset/YCB_Video_Dataset' \
    --train_list 'train_data.txt' --test_list 'test_data.txt'
