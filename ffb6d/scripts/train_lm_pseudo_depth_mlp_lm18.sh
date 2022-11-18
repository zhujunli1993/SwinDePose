#!/bin/bash
GPU_NUM=4
GPU_COUNT=1
NAME='lm_18_pseudo_depth_mlp_noSyn_phone_mulLR'
WANDB_PROJ='pose_estimation'
export CUDA_VISIBLE_DEVICES=$GPU_NUM
CLS='phone'
EXP_DIR='/workspace/REPO/pose_estimation/ffb6d/train_log'
LOG_EVAL_DIR="$EXP_DIR/$NAME/$CLS/eval_results"
SAVE_CHECKPOINT="$EXP_DIR/$NAME/$CLS/checkpoints"
LOG_TRAININFO_DIR="$EXP_DIR/$NAME/$CLS/train_info"
# checkpoint to resume. 
#tst_mdl="train_log/linemod_half_pseang_1/checkpoints/${cls}/FFB6D_${cls}_best.pth.tar"
python -m torch.distributed.launch --nproc_per_node=$GPU_COUNT --master_port 50001 apps/train_lm_pseudo_depth_lm18.py \
    --gpus=$GPU_COUNT \
    --wandb_proj $WANDB_PROJ \
    --wandb_name $NAME \
    --num_threads 4 \
    --gpu_id $GPU_NUM \
    --gpus $GPU_COUNT \
    --gpu '4' \
    --dataset_name 'linemod' \
    --data_root '/workspace/DATA/Linemod_preprocessed' \
    --train_list 'train.txt' --test_list 'test.txt' \
    --linemod_cls=$CLS \
    --lm_no_fuse --lm_no_render \
    --mini_batch_size 3 --val_mini_batch_size 3 \
    --lr 1e-5 \
    --add_depth --depth_split \
    --psp_out 256 --psp_size 256 --deep_features_size 256 \
    --ds_rgb_oc 64 128 256 256 \
    --ds_depth_oc 64 128 256 256  \
    --up_rgb_oc 256 64 64 \
    --log_eval_dir $LOG_EVAL_DIR --save_checkpoint $SAVE_CHECKPOINT --log_traininfo_dir $LOG_TRAININFO_DIR