#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

EXP_DIR=exps/singlebaseline

#mkdir ${EXP_DIR}
PY_ARGS=${@:1}
python -u main_single.py \
    --epochs 8 \
    --num_queries 300 \
    --batch_size 4 \
    --num_workers 8 \
    --lr_drop_epochs 6 7 \
    --dataset_file vid_single \
    --output_dir ${EXP_DIR} \
    --coco_pretrain \
    --coco_path ../../exp/Mask2Former/datasets/coco \
    --vid_path ../data/vid \
    --resume ../Pretrained/FAQ_ConditionalDETR_r50_epoch50.pth \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.train.$T

