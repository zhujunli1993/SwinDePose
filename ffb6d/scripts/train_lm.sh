#!/bin/bash
n_gpu=2
cls='phone'
#tst_mdl="train_log/linemod_half_pseang_1/checkpoints/${cls}/FFB6D_${cls}_best.pth.tar"
python3 -m torch.distributed.launch --master_port 29627 --nproc_per_node=$n_gpu train_lm.py --gpus=$n_gpu --cls=$cls 
