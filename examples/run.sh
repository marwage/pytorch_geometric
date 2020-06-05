#!/bin/bash

NAME="cluster_gcn_reddit_checkpoint"

git rev-parse HEAD >> ${NAME}.log

nvidia-smi dmon -s umt -o T -f ${NAME}.smi &

python3 ${NAME}.py >> ${NAME}.log

pkill -P $$
