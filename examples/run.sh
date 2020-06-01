#!/bin/bash

NAME="cluster_gcn_ppi"

nvidia-smi dmon -s umt -o T -f ${NAME}.smi &

python3 ${NAME}.py

pkill -P $$
