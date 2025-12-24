#!/bin/bash


#!/bin/bash

PYTHONPATH="."
LR=5e-3
DEVICE1="cuda:0"

TOP_K=(10)
DATASET_VERSION="twitter"

#BACKBONE="bpr"
num_cats=5
num_leaves=5
num_augs=3
#aug_ratio=0.8
sample_weight=5
BACKBONE="lightgcn"
grad_dim=512


(exec nohup env PYTHONPATH=${PYTHONPATH} \
    CUDA_LAUNCH_BLOCKING=1 \
    python model/main.py \
    --weight_path="weights/${BACKBONE}_backbone/${DATASET_VERSION}/aug${num_augs}_cat${num_cats}_leaf${num_leaves}_outweight${sample_weight}_lr${LR}_mask_train_@10" \
    --mask_train="True" \
    --top_k "${TOP_K[@]}"\
    --aug_gap=40 \
    --device=${DEVICE1} \
    --model=${BACKBONE} \
    --num_cats=${num_cats} \
    --num_leaves=${num_leaves} \
    --num_augs=${num_augs} \
    --sample_weight=${sample_weight} \
    --grad_dim=${grad_dim} \
    > logs/${BACKBONE}_backbone/${DATASET_VERSION}/aug${num_augs}_cat${num_cats}_leaf${num_leaves}_outweight${sample_weight}_lr_${LR}_mask_train_@10.log 2>&1 &)
    #    --aug_ratio=${aug_ratio} \
    #

