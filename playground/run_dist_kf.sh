#!/bin/bash

export KUNGFU_CONFIG_LOG_LEVEL=INFO # DEBUG | INFO | WARN | ERROR

rm /tmp/kungfu-run-*

kungfu-run \
    -np 4 \
    python3 benchmark_dist_kf.py \
        --dataset products

