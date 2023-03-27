#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

EXP_DIR=exps/multibaseline/
mkdir -p ${EXP_DIR}
PY_ARGS=${@:1}
python -u main_multi.py \
    --backbone resnet50 \
    --epochs 7 \
    --num_queries 300 \
    --batch_size 1 \
    --num_ref_frames 14 \
    --lr_drop_epochs 4 6 \
    --num_workers 16 \
    --dataset_file vid_multi \
    --output_dir ${EXP_DIR} \
    --coco_path ../../exp/Mask2Former/datasets/coco \
    --vid_path ../data/vid \
    --resume ./exps/singlebaseline/checkpoint.pth \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.train.$T


