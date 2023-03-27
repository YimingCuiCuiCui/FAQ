#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

EXP_DIR=exps/singlebaseline

#mkdir ${EXP_DIR}
PY_ARGS=${@:1}
python -u main_single.py \
    --epochs 8 \
    --num_feature_levels 4\
    --num_queries 300 \
    --batch_size 4 \
    --num_workers 8 \
    --lr_drop_epochs 6 7 \
    --with_box_refine \
    --dataset_file vid_single \
    --output_dir ${EXP_DIR} \
    --coco_pretrain \
    --coco_path ../../exp/Mask2Former/datasets/coco \
    --vid_path ../data/vid \
    --resume ../Pretrained/r50_faq_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.train.$T
