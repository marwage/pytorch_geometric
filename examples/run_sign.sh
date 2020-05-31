#!/bin/bash

trap "kill %1" SIGINT

nvidia-smi dmon -s umt -o T -f sign_reddit.smi &

python3 sign_reddit.py
