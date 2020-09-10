#!/bin/bash

NSIGHT_COMPUTE=/usr/local/cuda/bin/nv-nsight-cu-cli
PYTHON=/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3

MODEL=sage
DATASET=flickr

sudo $NSIGHT_COMPUTE \
	-o ${MODEL}_${DATASET}_chunk_cu \
	$PYTHON benchmark_sage_batch_act.py \
       		--model $MODEL \
		--dataset $DATASET \
		--adj

