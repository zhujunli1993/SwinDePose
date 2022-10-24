#!/bin/bash
GPU_NUM=4
GPU_COUNT=1
export CUDA_VISIBLE_DEVICES=$GPU_NUM
CLS='phone'
NAME='lm_rgb_offical'
WANDB_PROJ='pose_estimation'
EXP_DIR='/workspace/REPO/pose_estimation/ffb6d/train_log'
LOG_EVAL_DIR="$EXP_DIR/$NAME/$CLS/eval_results"
SAVE_CHECKPOINT="$EXP_DIR/$NAME/$CLS/checkpoints"
LOG_TRAININFO_DIR="$EXP_DIR/$NAME/$CLS/train_info"
tst_mdl="$SAVE_CHECKPOINT/FFB6D_$CLS.pth.tar"
python -m torch.distributed.launch --nproc_per_node=$GPU_COUNT --master_port 40000 apps/train_lm_pseudoOnly.py \
    --gpus=$GPU_COUNT \
    --wandb_proj $WANDB_PROJ \
    --wandb_name $NAME \
    --num_threads 0 \
    --dataset_name 'linemod' \
    --data_root '/workspace/DATA/Linemod_preprocessed' \
    --train_list 'train.txt' --test_list 'test.txt' \
    --linemod_cls=$CLS \
    --load_checkpoint $tst_mdl \
    --test --test_pose --eval_net \
    --pseudo_only \
    --test_gt \
    --log_eval_dir $LOG_EVAL_DIR --save_checkpoint $SAVE_CHECKPOINT --log_traininfo_dir $LOG_TRAININFO_DIR