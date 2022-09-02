#!/bin/bash
tst_mdl=train_log/ycb/checkpoints/FFB6D_best.pth.tar
python3 -m demo -checkpoint $tst_mdl -dataset ycb -log_eval_dir train_log/ycb/eval_results
