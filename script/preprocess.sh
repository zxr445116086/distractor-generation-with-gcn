#!/usr/bin/env bash

python -u preprocess.py \
        -train_dir=data/train.json \
        -valid_dir=data/dev.json \
        -save_data=data/processed \
        -share_vocab \
        -total_token_length=500 \
        -src_seq_length=60 \
        -src_sent_length=40 \
        -lower

