#!/bin/bash

NAME="ogbn_products_sage"

git rev-parse HEAD > ${NAME}.out

nvidia-smi dmon -s umt -o T -f ${NAME}.smi &

python3 ${NAME}.py >> ${NAME}.out

pkill -P $$
