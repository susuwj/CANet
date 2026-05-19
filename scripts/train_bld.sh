#!/usr/bin/env bash

MVS_TRAINING=""
LOG_DIR=""
GROUP_DIM="8,8,4,4,4"
HYPO_NUM="16,8,4,4,4"
DEPTH_INTER="0.5,0.5,0.5,0.5,1"
STAGE_LW="1,1,1,1,1"
QKV_DIM="4,4,4,4,4"
HEAD="4,4,4,4,4"
LEVELS=5
LREPOCHS="10,12,14:2"
N_VIEWS="5"
IMG_WH=640,512
GPU=0
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

CUDA_VISIBLE_DEVICES=$GPU python train.py --which_dataset="blendedmvs" \
                --epochs=16 \
                --logdir=$LOG_DIR \
                --trainpath=$MVS_TRAINING \
                --testpath=$MVS_TRAINING \
                --trainlist="datasets/lists/blendedmvs/low_res_all.txt" \
                --testlist="datasets/lists/blendedmvs/val.txt" \
                --n_views=$N_VIEWS \
                --batch_size=4 \
                --img_wh=$IMG_WH \
                --levels=$LEVELS \
                --group_cor_dim_stages=$GROUP_DIM \
                --hypo_plane_num_stages=$HYPO_NUM \
                --depth_interval_ratio_stages=$DEPTH_INTER \
                --qkv_dim_stages=$QKV_DIM \
                --heads_stages=$HEAD \
                --stage_lw=$STAGE_LW \
                --lr=0.001 \
                --lrepochs=$LREPOCHS \
                --robust_train


