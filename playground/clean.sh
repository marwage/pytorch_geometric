#!/bin/bash

rm *.log
rm *.smi
rm *.out

if [ $1 ]; then
    if [ $1 == "all" ]; then
	find . -name processed|xargs rm -r
    fi
fi

pkill nvidia-smi

