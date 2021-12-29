#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate pyg_cuda102
python -m classifier --train  --gpu 0 --val 5 --epoch 200 \
                     --dataset 'ModelNet40' \
                     --network 'DGCNNCls' --R 0.05 \
                     --batch 32 --worker 6 \
                     --lr 0.001 --weight_decay 0\
                     --lrd_factor 0.5 --lrd_step 20 \
                     --odir 'outputs'