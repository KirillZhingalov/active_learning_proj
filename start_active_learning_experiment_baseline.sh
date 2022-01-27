#!/bin/bash

python3 train_active_learning.py --experiment-name baseline --data-root ./data/EMNLP_dataset \
      --model-name bert-base-cased --num-pretraining-epochs 3 --pretraining-part 0.05 \
      --random-sample --sentence-max-len 128 --active-learning-lr 5e-5 --initial-lr 5e-5 \
      --train-batch-size 128 --val-batch-size 256 \
      --active-learning-new-samples-size 4 --num-active-learning-epochs 3
