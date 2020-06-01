#!/bin/bash

nvidia-smi dmon -s umt -o T -f cluster_gcn_reddit.smi &

python3 cluster_gcn_reddit.py

pkill -P $$
