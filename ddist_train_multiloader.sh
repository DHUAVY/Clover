#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES="0,1,3,4,5,6"
export OMP_NUM_THREADS=6

CONFIG='/data_sas/fhr/Clover/configs/exp_local/pretrain_msrvtt_cc3m.py'
GPUS=6
HOSTS=127.0.0.1
PORT=10001

echo "The distributed PORT is ${PORT}"

torchrun --nproc_per_node=$GPUS \
         --master_port=$PORT \
         --master_addr $HOSTS \
         --node_rank=0 \
         --nnodes=1 \
         train_multiloader.py $CONFIG --launcher pytorch ${@:7}
