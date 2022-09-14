#!/bin/bash
# checkpoint to resume.
# tst_mdl=train_log/ycb_all_real_newtrainlist_onlydata/checkpoints/FFB6D_best.pth.tar  
GPU_NUM=0
GPU_COUNT=1
NAME='vtesting'
WANDB_PROJ='pose_estimation'
export CUDA_VISIBLE_DEVICES=$GPU_NUM
CLS='phone'
#tst_mdl="train_log/linemod_half_pseang_1/checkpoints/${cls}/FFB6D_${cls}_best.pth.tar"
python -m torch.distributed.launch --nproc_per_node=$GPU_COUNT apps/train_lm_full.py \
    --gpus=$GPU_COUNT \
    --wandb_proj $WANDB_PROJ \
    --wandb_name $NAME \
    --num_threads 0 \
    --dataset_name 'linemod' \
    --data_root '/workspace/DATA/Linemod_preprocessed' \
    --train_list 'train.txt' --test_list 'test.txt' \
    --linemod_cls=$CLS 
