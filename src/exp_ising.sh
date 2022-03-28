#!/bin/sh

NAME_FUN="ising-2"
NUM_ITER=500

python run_bo_binary.py --function $NAME_FUN --surrogate ours --iteration $NUM_ITER
python run_bo_binary.py --function $NAME_FUN --surrogate gaussian_process --iteration $NUM_ITER
