#!/bin/sh

NAME_FUN="branin"
NUM_ITER=500

python run_bo.py --function $NAME_FUN --surrogate ours --iteration $NUM_ITER
python run_bo.py --function $NAME_FUN --surrogate gaussian_process --iteration $NUM_ITER
