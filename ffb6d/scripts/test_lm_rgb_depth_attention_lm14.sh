#!/bin/bash
GPU_NUM=1
GPU_COUNT=1
NAME='lm_14_rgb_depth_att_phone_new'
WANDB_PROJ='pose_estimation'
export CUDA_VISIBLE_DEVICES=$GPU_NUM
CLS='phone'
EXP_DIR='/workspace/REPO/pose_estimation/ffb6d/train_log'
LOG_EVAL_DIR="$EXP_DIR/$NAME/$CLS/eval_results"
SAVE_CHECKPOINT="$EXP_DIR/$NAME/$CLS/checkpoints"
LOG_TRAININFO_DIR="$EXP_DIR/$NAME/$CLS/train_info"
# checkpoint to resume. 
tst_mdl="$SAVE_CHECKPOINT/FFB6D_$CLS.pth.tar"
python -m torch.distributed.launch --nproc_per_node=$GPU_COUNT --master_port 60001 apps/train_lm_rgb_depth.py \
    --gpus=$GPU_COUNT \
    --wandb_proj $WANDB_PROJ \
    --wandb_name $NAME \
    --num_threads 1 \
    --gpu_id $GPU_NUM \
    --gpus $GPU_COUNT \
    --gpu '1' \
    --dataset_name 'linemod' \
    --data_root '/workspace/DATA/Linemod_preprocessed' \
    --train_list 'train.txt' --test_list 'test.txt' \
    --load_checkpoint $tst_mdl \
    --mini_batch_size 3 --val_mini_batch_size 3 --test_mini_batch_size 1 \
    --linemod_cls=$CLS \
    --test --test_pose --eval_net \
    --test_gt \
    --attention --rgb_only --depth_only \
    --lm_no_fuse --lm_no_render \
    --psp_out 1024 --psp_size 512 --deep_features_size 256 \
    --ds_rgb_oc 64 128 512 1024 \
    --ds_depth_oc_fuse 128 256 512 512 \
    --ds_depth_oc 64 128 256 512 \
    --up_depth_oc 512 256 64 64 \
    --up_rgb_oc 256 64 64 \
    --log_eval_dir $LOG_EVAL_DIR --save_checkpoint $SAVE_CHECKPOINT --log_traininfo_dir $LOG_TRAININFO_DIR \
