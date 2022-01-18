#!/bin/bash

python3 train_active_learning.py --experiment-name baseline --data-root ./data/EMNLP_dataset --model-name bert-base-cased --num-pretraining-epochs 3 --pretraining-part 0.1 --random-sample --sentence-max-len 128 --active-learning-lr 5e-5 --initial-lr 5e-5
