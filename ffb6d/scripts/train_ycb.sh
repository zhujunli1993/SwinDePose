#!/bin/bash
GPU_NUM=4
GPU_COUNT=1
NAME='vtesting'
WANDB_PROJ='pose_estimation'
export CUDA_VISIBLE_DEVICES=$GPU_NUM
EXP_DIR='/workspace/REPO/pose_estimation/ffb6d/train_log'
LOG_EVAL_DIR="$EXP_DIR/$NAME/ycb/eval_results"
SAVE_CHECKPOINT="$EXP_DIR/$NAME/ycb/checkpoints"
LOG_TRAININFO_DIR="$EXP_DIR/$NAME/ycb/train_info"
python -m torch.distributed.launch --nproc_per_node=$GPU_COUNT apps/train_ycb.py \
    --gpus=$GPU_COUNT \
    --wandb_proj $WANDB_PROJ \
    --wandb_name $NAME \
    --num_threads 0 \
    --data_root '/workspace/DATA/YCBV' \
    --train_list 'train_data_list.txt' --test_list 'test_data.txt' \
    --gpu_id $GPU_NUM \
    --gpus $GPU_COUNT \
    --gpu '0,3,6,7' \
    --lr 1e-2 --in_c 9 \
    --dataset_name 'ycb' \
    --mini_batch_size 1 --val_mini_batch_size 1 \
    --log_eval_dir $LOG_EVAL_DIR --save_checkpoint $SAVE_CHECKPOINT --log_traininfo_dir $LOG_TRAININFO_DIR

