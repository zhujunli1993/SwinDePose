#!/bin/bash
GPU_NUM=7
GPU_COUNT=1
NAME='lm_10_pseudo_noSyn_depth_attention_new'
WANDB_PROJ='pose_estimation'
export CUDA_VISIBLE_DEVICES=$GPU_NUM
CLS='phone'
EXP_DIR='/workspace/REPO/pose_estimation/ffb6d/train_log'
LOG_EVAL_DIR="$EXP_DIR/$NAME/$CLS/eval_results"
SAVE_CHECKPOINT="$EXP_DIR/$NAME/$CLS/checkpoints"
LOG_TRAININFO_DIR="$EXP_DIR/$NAME/$CLS/train_info"
# checkpoint to resume. 
tst_mdl="$SAVE_CHECKPOINT/FFB6D_$CLS.pth.tar"
python -m torch.distributed.launch --nproc_per_node=$GPU_COUNT --master_port 50002 apps/train_lm_pseudo_depth_lm10.py \
    --gpus=$GPU_COUNT \
    --wandb_proj $WANDB_PROJ \
    --wandb_name $NAME \
    --num_threads 1 \
    --gpu_id $GPU_NUM \
    --gpus $GPU_COUNT \
    --gpu '7' \
    --dataset_name 'linemod' \
    --load_checkpoint $tst_mdl \
    --test --test_pose --eval_net \
    --test_gt \
    --data_root '/workspace/DATA/Linemod_preprocessed' \
    --train_list 'train.txt' --test_list 'test.txt' \
    --linemod_cls=$CLS \
    --attention \
    --mini_batch_size 3 --val_mini_batch_size 3 --test_mini_batch_size 1 \
    --add_depth --depth_split \
    --psp_out 1024 --psp_size 512 --deep_features_size 256 \
    --ds_rgb_oc 64 128 512 1024 \
    --ds_depth_oc_fuse 128 256 512 512 \
    --ds_depth_oc 64 128 256 512 \
    --up_depth_oc 512 256 64 64 \
    --up_rgb_oc 256 64 64 \
    --log_eval_dir $LOG_EVAL_DIR --save_checkpoint $SAVE_CHECKPOINT --log_traininfo_dir $LOG_TRAININFO_DIR