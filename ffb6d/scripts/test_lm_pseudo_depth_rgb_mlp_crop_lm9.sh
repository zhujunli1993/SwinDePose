#!/bin/bash
GPU_NUM=4
GPU_COUNT=1
NAME='lm_9_pseudo_noSyn_depth_RGB_mlp'
WANDB_PROJ='pose_estimation'
export CUDA_VISIBLE_DEVICES=$GPU_NUM
CLS='phone'
EXP_DIR='/workspace/REPO/pose_estimation/ffb6d/train_log'
LOG_EVAL_DIR="$EXP_DIR/$NAME/$CLS/eval_results"
SAVE_CHECKPOINT="$EXP_DIR/$NAME/$CLS/checkpoints"
LOG_TRAININFO_DIR="$EXP_DIR/$NAME/$CLS/train_info"
# checkpoint to resume. 
tst_mdl="$SAVE_CHECKPOINT/FFB6D_$CLS.pth.tar"
python -m torch.distributed.launch --nproc_per_node=$GPU_COUNT --master_port 50001 apps/train_lm_pseudo_depth_rgb_lm9.py \
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
    --test_gt \
    --n_total_epoch 30 \
    --lm_no_fuse --lm_no_render --add_depth --depth_split \
    --add_rgb \
    --psp_out 1024 --psp_size 1024 --deep_features_size 256 \
    --ds_rgb_oc 64 128 512 1024 \
    --ds_rgb_oc_ori 64 128 512 1024 \
    --ds_rgb_ori_oc_fuse 64 128 512 1024 \
    --ds_depth_oc_fuse 64 128 512 1024 \
    --up_depth_oc 256 64 64 \
    --up_rgb_oc 256 64 64 \
    --crop --width 320 --height 240 --n_sample_points 6400 --max_w 320 --max_h 240 \
    --log_eval_dir $LOG_EVAL_DIR --save_checkpoint $SAVE_CHECKPOINT --log_traininfo_dir $LOG_TRAININFO_DIR \