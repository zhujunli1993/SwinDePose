#!/bin/bash
GPU_NUM=0
GPU_COUNT=1
NAME='ycbv_swinTiny_fullSyn_dense_fullInc'
WANDB_PROJ='pose_estimation'
export CUDA_VISIBLE_DEVICES=$GPU_NUM
EXP_DIR='/workspace/REPO/pose_estimation/ffb6d/train_log'
LOG_EVAL_DIR="$EXP_DIR/$NAME/ycb/eval_results"
SAVE_CHECKPOINT="$EXP_DIR/$NAME/ycb/checkpoints"
LOG_TRAININFO_DIR="$EXP_DIR/$NAME/ycb/train_info"
# checkpoint to resume. 
tst_mdl="$SAVE_CHECKPOINT/FFB6D_ycb.pth.tar"
python -m torch.distributed.launch --nproc_per_node=$GPU_COUNT --master_port 60088 apps/train_ycb.py \
    --gpus=$GPU_COUNT \
    --wandb_proj $WANDB_PROJ \
    --wandb_name $NAME \
    --num_threads 4 \
    --gpu_id $GPU_NUM \
    --gpus $GPU_COUNT \
    --lr 1e-2 --cyc_max_lr 1e-3 --cyc_base_lr 1e-5 --clr_div 9 \
    --data_root '/workspace/DATA/YCB_Video_Dataset' \
    --train_list 'train_data.txt' --test_list 'test_data.txt' \
    --gpu_id $GPU_NUM \
    --gpus $GPU_COUNT \
    --dataset_name 'ycb' \
    --in_c 9 \
    --load_checkpoint $tst_mdl \
    --test --test_pose --eval_net \
    --mini_batch_size 3 --val_mini_batch_size 3 --test_mini_batch_size 1 \
    --log_eval_dir $LOG_EVAL_DIR --save_checkpoint $SAVE_CHECKPOINT --log_traininfo_dir $LOG_TRAININFO_DIR