#!/bin/bash

NAME="cluster_gcn_reddit_f1"

git rev-parse HEAD > ${NAME}.out

nvidia-smi dmon -s umt -o T -f ${NAME}.smi &

python3 ${NAME}.py >> ${NAME}.out

pkill -P $$
