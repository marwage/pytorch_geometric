#!/bin/bash

nvidia-smi dmon -s umt -o T -f sign_reddit.txt &

python3 sign_reddit.py
