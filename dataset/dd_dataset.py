import os
import torch
import numpy as np
import typing as tp
import random as rnd
from torch.utils.data import Dataset

from transformers import BertTokenizerFast


class DailyDialogDataset(Dataset):
    def __init__(self, root: str, subset: str = 'train', model_name: str = "bert-base-uncased",
                 sent_max_len: int = 512):
        assert subset in ['train', 'validation', 'test']

        self.root = root
        self.subset = subset

        self.model_name = model_name
        # max sequence length for each document/sentence sample
        self.max_length = sent_max_len
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name, do_lower_case=True)

        self.encodings = self.load_encodings(os.path.join(root, subset, f'dialogues_{subset}.txt'))
        self.labels = self.load_labels(os.path.join(root, subset, f'dialogues_act_{subset}.txt'))

        self.sampling_class = None

    def __len__(self):
        return len(self.labels)

    def load_encodings(self, path: str):
        print(f'Loading and preprocessing texts from {path}...')
        with open(path) as fp:
            samples = fp.readlines()

        texts = []
        for text in samples:
            for phrase in text.replace('\n', '').split('__eou__'):
                if not len(phrase): continue
                texts.append(phrase)

        # set truncation to True so that we eliminate tokens that go above max_length
        # set padding to True to pad documents that are less than max_length with empty tokens
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_length)
        return encodings

    def load_labels(self, path: str) -> np.ndarray:
        print(f'Loading labels from {path}...')
        with open(path) as fp:
            labels = fp.readlines()

        targets = []
        for text_labels in labels:
            targets.extend(list(map(int, text_labels.replace('\n', '').strip().split(' '))))
        return np.array(targets)

    def get_batch_with_class(self, class_idx: int, batch_size: int = 1):
        # Get indexes of samples with class == class_idx
        class_indexes = np.argwhere(self.labels == class_idx).reshape(-1)
        # Sample batch_size of samples indexes
        batch_indexes = rnd.sample(class_indexes.tolist(), batch_size)

        # Creating batch
        batch = None
        for index in batch_indexes:
            if batch is None: batch = self[index]
            else:
                for k, v in self[index].items():
                    batch[k] = torch.stack([batch[k], v])
        return batch

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item


if __name__ == '__main__':
    dataset = DailyDialogDataset('/home/kirill/MDS/active_learning_proj/data/EMNLP_dataset',
                                 subset='validation')

    batch = dataset.get_batch_with_class(class_idx=1, batch_size=2)

    for k, v in batch.items():
        print(k, v.shape)
    print(batch['labels'])

    print('='*80)

    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=2)
    for sample in loader:
        for k, v in sample.items():
            print(k, v.shape)
        print(sample['labels'])
        break
