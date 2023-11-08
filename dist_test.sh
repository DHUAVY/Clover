#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0,1,3,4,5,6"
export BYTED_TORCH_FX=O0
export OMP_NUM_THREADS=6

CONFIG="/data_sas/fhr/Clover/configs/exp_local/finetune_msrvtt_retrieval.py"
CHECKPOINT="/data_sas/fhr/Clover/work_dirs/pretrain_msrvtt_cc3m/latest.pth"
GPUS=6
PORT=10001

torchrun   \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    test.py $CONFIG $CHECKPOINT \
    --out /data_sas/fhr/Clover/work_dirs/finetune_msrvtt_retrieval/test_result/result.json \
    --eval recall_for_video_text_retrieval \
    --launcher pytorch ${@:4}
