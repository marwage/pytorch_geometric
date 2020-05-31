#!/bin/bash

nvidia-smi dmon -s umt -o T -f graph_saint_reddit.smi &

python3 graph_saint_reddit.py

pkill -P $$
