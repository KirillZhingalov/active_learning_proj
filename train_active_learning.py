import os
import argparse as ap
from tqdm import tqdm

from transformers import BertForSequenceClassification
from transformers import AdamW, get_scheduler

from utils.metrics import Meter
from dataset.dd_dataset import DailyDialogDataset
import torch
from torch.utils.data import DataLoader


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
    parser.add_argument('--active-learning-batch-size', type=int, default=2)
    parser.add_argument('--train-batch-size', type=int, default=8)
    parser.add_argument('--val-batch-size', type=int, default=24)

    parser.add_argument('--initial-lr', type=float, default=5e-5)
    parser.add_argument('--warmup-steps', type=int, default=100)

    return parser.parse_args()


def train_step():
    model.train()
    for iter_idx, sample in tqdm(enumerate(train_loader), total=len(train_loader)):
        input_ids = sample['input_ids'].to(torch.device('cuda:0'))
        token_type_ids = sample['token_type_ids'].to(torch.device('cuda:0'))
        attention_mask = sample['token_type_ids'].to(torch.device('cuda:0'))
        labels = sample['labels'].to(torch.device('cuda:0'))

        out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                    labels=labels)

        loss = out.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if iter_idx == 10: break


def validation_step():
    meter = Meter()
    model.eval()
    for iter_idx, sample in tqdm(enumerate(val_loader), total=len(val_loader)):
        input_ids = sample['input_ids'].to(torch.device('cuda:0'))
        token_type_ids = sample['token_type_ids'].to(torch.device('cuda:0'))
        attention_mask = sample['token_type_ids'].to(torch.device('cuda:0'))
        labels = sample['labels'].to(torch.device('cuda:0'))

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                        labels=labels)

        logits = out.logits
        predictions = torch.argmax(logits, dim=-1)
        meter.update(predictions, labels)
        if iter_idx == 10: break
    return meter.get_report()


def get_cls_idx_with_worst_metric(validation_metrics):
    if validation_metrics is None: return 1
    else:
        min_f1, cls_idx = float('inf'), -1
        for idx in range(5):
            if validation_metrics[idx]['f1-score'] < min_f1:
                min_f1 = validation_metrics[idx]['f1-score']
                cls_idx = idx
    return cls_idx


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(os.path.join(args.logs_dir, args.experiment_name)):
        os.makedirs(os.path.join(args.logs_dir, args.experiment_name))

    if not os.path.exists(os.path.join(args.checkpoint_dir, args.experiment_name)):
        os.makedirs(os.path.join(args.checkpoint_dir, args.experiment_name))

    # Creating model
    print('Creating model...')
    model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=5)
    model.to(torch.device('cuda:0'))
    print('Model created')
    print(model)

    # Creating datasets
    train_dataset = DailyDialogDataset(root=args.data_root, subset='train', model_name=args.model_name,
                                       sent_max_len=args.sentence_max_len)
    val_dataset = DailyDialogDataset(root=args.data_root, subset='validation', model_name=args.model_name,
                                     sent_max_len=args.sentence_max_len)

    train_loader = DataLoader(train_dataset, args.train_batch_size)
    val_loader = DataLoader(val_dataset, args.val_batch_size)

    # Creating optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.initial_lr)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.num_epochs * len(train_loader)
    )

    # Pretraining with part of training data for few epochs
    print(f'Start pretraining for {args.num_pretraining_epochs} epochs')
    meter = Meter()
    validation_metrics = None
    for epoch_num in range(args.num_pretraining_epochs):
        train_step()
        validation_metrics = validation_step()
        print(validation_metrics)

    print('Classification report after pretraining:')
    print(meter.get_report())

    # Active learning stage
    print('Starting active learning stage')
    meter = Meter()
    for iter_num in range(args.num_iterations):
        # Get class index with worst F1-score
        cls_idx = get_cls_idx_with_worst_metric(validation_metrics)

        # Get samples with class cls_idx
        sample = train_dataset.get_batch_with_class(cls_idx, args.active_learning_batch_size)
        input_ids = sample['input_ids'].to(torch.device('cuda:0'))
        token_type_ids = sample['token_type_ids'].to(torch.device('cuda:0'))
        attention_mask = sample['token_type_ids'].to(torch.device('cuda:0'))
        labels = sample['labels'].to(torch.device('cuda:0'))

        # fine-tune model
        out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                    labels=labels)
        loss = out.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Compute metrics
        validation_metrics = validation_step()

    print('Classification report after active learning stage:')
    print(meter.get_report())

