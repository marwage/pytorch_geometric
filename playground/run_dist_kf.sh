#!/bin/bash

export KUNGFU_CONFIG_LOG_LEVEL=DEBUG # DEBUG | INFO | WARN | ERROR

kungfu-run \
    -np 2 \
    python3 benchmark_dist_kf.py \
        --dataset flickr

