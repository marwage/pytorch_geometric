#!/bin/bash

rm *.log
rm *.smi
rm *.out

if [ $1 ]; then
    if [ $1 == "all" ]; then
        rm -r ../data/*/processed
    fi
fi
