#!/bin/bash
GPU_NUM=6
GPU_COUNT=1
export CUDA_VISIBLE_DEVICES=$GPU_NUM
CLS='cam'
SAVE_CHECKPOINT="train_log/final_time/cam/checkpoints/cam.pth.tar"
# checkpoint to resume. 
tst_mdl=$SAVE_CHECKPOINT
python -m torch.distributed.launch --nproc_per_node=$GPU_COUNT --master_port 60017 apps/infer_lab.py \
    --gpus=$GPU_COUNT \
    --num_threads 0 \
    --gpu_id $GPU_NUM \
    --gpus $GPU_COUNT \
    --gpu '0,3,6,7' \
    --lr 1e-2 \
    --dataset_name 'lab' \
    --data_root '/workspace/DATA/LabROS' \
    --lab_depth_input '/workspace/DATA/LabROS/data/all_depth.npy' \
    --lab_pose_save '/workspace/DATA/LabROS' \
    --linemod_cls $CLS \
    --in_c 9 --lm_no_pbr \
    --load_checkpoint $tst_mdl \
    --test --test_pose --eval_net \
    --mini_batch_size 3 --val_mini_batch_size 3 --test_mini_batch_size 1 
