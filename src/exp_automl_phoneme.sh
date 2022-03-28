#!/bin/sh

NAME_DATASET="phoneme"
NUM_ITER=100

python run_bo_automl.py --dataset $NAME_DATASET --surrogate ours --iteration $NUM_ITER
python run_bo_automl.py --dataset $NAME_DATASET --surrogate gaussian_process --iteration $NUM_ITER
