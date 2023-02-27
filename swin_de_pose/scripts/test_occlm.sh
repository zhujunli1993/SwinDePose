#!/bin/bash
GPU_NUM=6
GPU_COUNT=1
CLS='glue'
NAME="occlm_swinTiny_"$CLS"_fullSyn_dense_fullInc"
WANDB_PROJ='pose_estimation'
export CUDA_VISIBLE_DEVICES=$GPU_NUM
EXP_DIR='/workspace/REPO/pose_estimation/ffb6d/train_log'
LOG_EVAL_DIR="$EXP_DIR/$NAME/$CLS/eval_results"
SAVE_CHECKPOINT="$EXP_DIR/$NAME/$CLS/checkpoints"
LOG_TRAININFO_DIR="$EXP_DIR/$NAME/$CLS/train_info"
# checkpoint to resume. 
tst_mdl="$SAVE_CHECKPOINT/FFB6D_$CLS.pth.tar"
python -m torch.distributed.launch --nproc_per_node=$GPU_COUNT --master_port 60008 apps/train_occlm.py \
    --gpus=$GPU_COUNT \
    --wandb_proj $WANDB_PROJ \
    --wandb_name $NAME \
    --num_threads 4 \
    --gpu_id $GPU_NUM \
    --gpus $GPU_COUNT \
    --gpu '0,3,6,7' \
    --lr 1e-2 \
    --dataset_name 'occlusion_linemod' \
    --data_root '/workspace/DATA/Occ_LineMod' \
    --load_checkpoint $tst_mdl \
    --train_list "train_pbr/training_msk_"$CLS".txt" --test_list "test/testing_msk_"$CLS".txt" \
    --occ_linemod_cls=$CLS \
    --in_c 9 \
    --test --test_pose --eval_net \
    --mini_batch_size 9 --val_mini_batch_size 9 --test_mini_batch_size 1 \
    --log_eval_dir $LOG_EVAL_DIR --save_checkpoint $SAVE_CHECKPOINT --log_traininfo_dir $LOG_TRAININFO_DIR
