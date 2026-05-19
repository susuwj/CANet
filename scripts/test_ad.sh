#!/usr/bin/env bash
LOG_DIR=""  
OUT_DIR="./outputs/tnt/"
GROUP_DIM="8,8,4,4,4"
HYPO_NUM="32,8,4,4,4"
DEPTH_INTER="0.5,0.5,0.5,0.5,1"
QKV_DIM="4,4,4,4,4"
HEAD="4,4,4,4,4"
LEVELS=5
TEST_ROOT=""
L_SCAN_NAME=( 'Auditorium' 'Ballroom' 'Courtroom' 'Museum' 'Palace' 'Temple' )
L_SCAN=$(seq 0 5)
for S in ${L_SCAN[@]}; do
  CUDA_VISIBLE_DEVICES=0 python test.py --nolog \
                                        --which_dataset="advanced" \
                                        --loadckpt=$LOG_DIR \
                                        --batch_size=1 \
                                        --outdir=$OUT_DIR \
                                        --logdir=$LOG_DIR  \
                                        --img_wh="-1, 1024"  \
                                        --levels=$LEVELS \
                                        --group_cor_dim_stages=$GROUP_DIM \
                                        --hypo_plane_num_stages=$HYPO_NUM \
                                        --depth_interval_ratio_stages=$DEPTH_INTER \
                                        --qkv_dim_stages=$QKV_DIM \
                                        --heads_stages=$HEAD \
                                        --testpath=$TEST_ROOT \
                                        --testlist=${L_SCAN_NAME[S]} \
                                        --n_views="11"
done
