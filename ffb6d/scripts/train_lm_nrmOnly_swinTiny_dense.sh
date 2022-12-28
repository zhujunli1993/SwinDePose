#!/bin/bash
GPU_NUM=0
GPU_COUNT=1
NAME='lm_swinTiny_driller_fullSyn_dense_fullInc'
WANDB_PROJ='pose_estimation'
export CUDA_VISIBLE_DEVICES=$GPU_NUM
CLS='driller'
EXP_DIR='/workspace/REPO/pose_estimation/ffb6d/train_log'
LOG_EVAL_DIR="$EXP_DIR/$NAME/$CLS/eval_results"
SAVE_CHECKPOINT="$EXP_DIR/$NAME/$CLS/checkpoints"
LOG_TRAININFO_DIR="$EXP_DIR/$NAME/$CLS/train_info"
# # checkpoint to resume. 
# tst_mdl="$SAVE_CHECKPOINT/FFB6D_$CLS.pth.tar"
python -m torch.distributed.launch --nproc_per_node=$GPU_COUNT --master_port 60098 apps/train_lm_nrmOnly_swinTiny_dense.py \
    --gpus=$GPU_COUNT \
    --wandb_proj $WANDB_PROJ \
    --wandb_name $NAME \
    --num_threads 4 \
    --gpu_id $GPU_NUM \
    --gpus $GPU_COUNT \
    --gpu '0,3,6,7' \
    --lr 1e-2 \
    --dataset_name 'linemod' \
    --data_root '/workspace/DATA/Linemod_preprocessed' \
    --train_list 'train.txt' --test_list 'test.txt' \
    --linemod_cls=$CLS \
    --in_c 9 --lm_no_pbr \
    --mini_batch_size 9 --val_mini_batch_size 9 \
    --log_eval_dir $LOG_EVAL_DIR --save_checkpoint $SAVE_CHECKPOINT --log_traininfo_dir $LOG_TRAININFO_DIR
