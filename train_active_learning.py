import os
import json
import numpy as np
import random as rnd
import argparse as ap
from tqdm import tqdm
from pprint import pprint
from collections import Counter

from sklearn.model_selection import train_test_split

from transformers import BertForSequenceClassification
from transformers import AdamW, get_scheduler

from utils.metrics import Meter
from dataset.dd_dataset import DailyDialogDataset
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = ap.ArgumentParser()

    # Required arguments
    parser.add_argument('--experiment-name', required=True, type=str)
    parser.add_argument('--data-root', required=True, type=str)
    parser.add_argument('--model-name', required=True, type=str)
    parser.add_argument('--random-sample', action='store_true',
                        help='Is batch sampled randomly during active learning stage')

    parser.add_argument('--logs-dir', type=str, default='./logs')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')

    parser.add_argument('--num-iterations', type=int, default=10000)
    parser.add_argument('--pretraining-part', type=float, default=0.1,
                        help='Percentage of training data for initial pretraining')

    parser.add_argument('--num-pretraining-epochs', type=int, default=10)
    parser.add_argument('--active-learning-new-samples-size', type=int, default=4)
    parser.add_argument('--num-active-learning-epochs', type=int, default=3)
    parser.add_argument('--train-batch-size', type=int, default=16)
    parser.add_argument('--val-batch-size', type=int, default=48)

    parser.add_argument('--sentence-max-len', type=int, default=512)
    parser.add_argument('--initial-lr', type=float, default=5e-5)
    parser.add_argument('--active-learning-lr', type=float, default=5e-7)
    parser.add_argument('--warmup-steps', type=int, default=100)

    return parser.parse_args()


def train_step(model, loader, optimizer, lr_scheduler):
    model.train()
    for iter_idx, sample in tqdm(enumerate(loader), total=len(loader)):
        for k, v in sample.items(): sample[k] = v.to(torch.device('cuda:0'))

        out = model(**sample)

        loss = out.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()


def validation_step(model, loader):
    meter = Meter()
    model.eval()
    for iter_idx, sample in tqdm(enumerate(loader), total=len(loader)):
        for k, v in sample.items(): sample[k] = v.to(torch.device('cuda:0'))

        with torch.no_grad():
            out = model(**sample)

        logits = out.logits
        predictions = torch.argmax(logits, dim=-1)
        meter.update(predictions, sample['labels'])

    return meter.get_report()


def get_cls_idx_with_worst_metric(validation_metrics):
    if validation_metrics is None: return 1
    else:
        min_f1, cls_idx = float('inf'), -1
        for idx in range(4):
            if validation_metrics[idx]['f1-score'] < min_f1:
                min_f1 = validation_metrics[idx]['f1-score']
                cls_idx = idx
    return cls_idx


if __name__ == '__main__':
    rnd.seed(42)  # Set random state seed
    args = parse_args()
    print(args)
    if not os.path.exists(os.path.join(args.logs_dir, args.experiment_name)):
        os.makedirs(os.path.join(args.logs_dir, args.experiment_name))

    if not os.path.exists(os.path.join(args.checkpoint_dir, args.experiment_name)):
        os.makedirs(os.path.join(args.checkpoint_dir, args.experiment_name))

    tb_logger = SummaryWriter(os.path.join(args.logs_dir, args.experiment_name))

    # Creating model
    print('Creating model...')
    model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=4)
    model.to(torch.device('cuda:0'))

    # Creating datasets
    full_train_dataset = DailyDialogDataset(root=args.data_root, subset='train', model_name=args.model_name,
                                       sent_max_len=args.sentence_max_len)
    val_dataset = DailyDialogDataset(root=args.data_root, subset='test', model_name=args.model_name,
                                     sent_max_len=args.sentence_max_len)

    # Making stratified pretraining loader
    train_targets = full_train_dataset.labels
    targets_indexes = {idx: [] for idx in range(4)}
    for idx, label in enumerate(train_targets):
        targets_indexes[label].append(idx)
    train_idxs = []
    for label in range(4):
        train_idxs.extend(rnd.sample(targets_indexes[label], int(args.pretraining_part * len(targets_indexes[label]))))

    train_dataset = torch.utils.data.Subset(full_train_dataset, indices=train_idxs)
    train_loader = DataLoader(train_dataset, args.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, args.val_batch_size)

    # Creating optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.initial_lr)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.num_pretraining_epochs * len(train_loader)
    )

    # Pretraining with part of training data for few epochs
    print(f'Start pretraining for {args.num_pretraining_epochs} epochs')
    validation_metrics = None
    for epoch_num in range(args.num_pretraining_epochs):
        train_step(model=model, loader=train_loader, optimizer=optimizer, lr_scheduler=lr_scheduler)
        validation_metrics = validation_step(model=model, loader=val_loader)
        pprint(validation_metrics)

        # Saving pretraining metrics
        for cls_idx in range(4):
            tb_logger.add_scalar(f'Pretrain/f1_class_{cls_idx}', validation_metrics[cls_idx]['f1-score'], epoch_num)
    # Saving pretrained model
    torch.save(model, os.path.join(args.checkpoint_dir, args.experiment_name, 'pretrained_model.pth'))

    # Active learning stage
    # 1. Get class with worst metric (or randomly sampling in baseline case)
    # 2. Add args.active_learning_new_samples_size samples of class from pt.1
    # 3. Train args.num_active_learning_epochs
    # 4. Repeat 1-3 until samples exists
    # 5. Calculate final metrics

    optimizer = AdamW(model.parameters(), lr=args.active_learning_lr)
    results = {idx: [] for idx in range(4)}

    for iter_num in range(100000000):
        print(f'Start iteration {iter_num}')
        if args.random_sample or validation_metrics is None:
            # Sample class number randomly
            cls_idx = rnd.randint(0, 3)
        else:
            # Get class with worst F1 score
            cls_idx = get_cls_idx_with_worst_metric(validation_metrics)

        # Add few samples of cls_idx in dataset
        if len(targets_indexes[cls_idx]) < args.active_learning_new_samples_size:
            # If not enough new samples in dataset, then exit
            break
        else:
            # Add samples and create new dataset
            train_idxs += rnd.sample(targets_indexes[cls_idx], args.active_learning_new_samples_size)
            train_dataset = torch.utils.data.Subset(full_train_dataset, indices=train_idxs)
            train_loader = DataLoader(train_dataset, args.train_batch_size, shuffle=True)

        print("Classes distribution:")
        pprint(Counter(train_dataset.dataset.labels))

        for epoch_num in range(args.num_active_learning_epochs):
            train_step(model=model, loader=train_loader, optimizer=optimizer, lr_scheduler=lr_scheduler)
            validation_metrics = validation_step(model=model, loader=val_loader)
        pprint(validation_metrics)

        # Saving pretraining metrics
        for cls_idx in range(4):
            tb_logger.add_scalar(f'ActiveLearning/f1_class_{cls_idx}',
                                 validation_metrics[cls_idx]['f1-score'],
                                 iter_num)
        tb_logger.flush()

        # Saving pretrained model
        torch.save(model, os.path.join(args.checkpoint_dir,
                                       args.experiment_name,
                                       f'active_learning_iter_{iter_num}.pth'))
