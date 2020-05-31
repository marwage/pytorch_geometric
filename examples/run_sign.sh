#!/bin/bash

nvidia-smi dmon -s umt -o T -f sign_reddit.smi &

python3 sign_reddit.py

pkill -P $$
