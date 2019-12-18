#!/usr/bin/env bash

PROJ=/home/zhouxiaorui/gcn_hred_directed_graph

export CUDA_VISIBLE_DEVICES=$1
FULL_MODEL_NAME=$2

python -u translate.py \
    -model=${PROJ}/data/model/${FULL_MODEL_NAME}.pt \
    -data=${PROJ}/data/test.json \
    -output=${PROJ}/data/pred/${FULL_MODEL_NAME}.txt \
    -share_vocab \
    -block_ngram_repeat=1 \
    -replace_unk \
    -batch_size=1 \
    -beam_size=10 \
    -n_best=10 \
    -gpu=0
