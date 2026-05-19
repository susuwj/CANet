#!/usr/bin/env bash
LOG_DIR="./checkpoints/"
OUT_DIR="./outputs/dtu/"
GROUP_DIM="8,8,4,4,4"
HYPO_NUM="16,8,4,4,4"
DEPTH_INTER="0.5,0.5,0.5,0.5,1"
QKV_DIM="4,4,4,4,4"
HEAD="4,4,4,4,4"
LEVELS=5
TEST_ROOT=""
L_SCAN_NAME=( 'scan1' 'scan4' 'scan9' 'scan10' 'scan11' 'scan12' 'scan13' 'scan15'
        'scan23' 'scan24' 'scan29' 'scan32' 'scan33' 'scan34' 'scan48' 'scan49'
        'scan62' 'scan75' 'scan77' 'scan110' 'scan114' 'scan118' )
L_SCAN=$(seq 0 21)
for S in ${L_SCAN[@]}; do
  CUDA_VISIBLE_DEVICES=0 python test.py --nolog \
                                      --which_dataset="dtu" \
                                      --loadckpt=$LOG_DIR \
                                      --batch_size=1 \
                                      --outdir=$OUT_DIR \
                                      --logdir=$LOG_DIR  \
                                      --img_wh="1600,1152"  \
                                      --levels=$LEVELS \
                                      --group_cor_dim_stages=$GROUP_DIM \
                                      --hypo_plane_num_stages=$HYPO_NUM \
                                      --depth_interval_ratio_stages=$DEPTH_INTER \
                                      --qkv_dim_stages=$QKV_DIM \
                                      --heads_stages=$HEAD \
                                      --testpath=$TEST_ROOT \
                                      --testlist=${L_SCAN_NAME[S]} \
                                      --n_views="5"
done
