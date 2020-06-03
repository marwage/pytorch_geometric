#!/bin/bash

NAME="graph_saint_reddit"

nvidia-smi dmon -s umt -o T -f ${NAME}.smi &

python3 ${NAME}.py

pkill -P $$
