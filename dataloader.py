import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import tiktoken
import numpy as np

import torch
from torch.utils.data import Dataset, random_split, DataLoader
from transformers import BertTokenizer

max_input_length = 1024
max_target_length = 1024

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


class ChnSentiCorpDataset(Dataset):

    def __init__(self, data_file=None, max_dataset_size=10000):
        assert data_file != None
        self.max_dataset_size = max_dataset_size
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        Data = {}
        with open(data_file, "rt") as f:
            for idx, line in enumerate(f):
                if idx >= self.max_dataset_size:
                    break
                if idx == 0:
                    continue
                label, text = int(line[0]), line[2:].strip()
                index = idx - 1
                Data[index] = {"label": label, "text": text}
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collote_fn(batch_samples):
    batch_inputs, batch_targets = [], []
    for sample in batch_samples:
        batch_inputs.append(sample["text"])
        batch_targets.append(sample["label"])
    batch_data = tokenizer(
        batch_inputs,
        padding=True,
        max_length=max_input_length,
        truncation=True,
        return_tensors="pt",
    )
    batch_data["labels"] = batch_targets
    return batch_data


# train_dataset = ChnSentiCorpDataset(data_file="dataset/train.csv")
# val_dataset = ChnSentiCorpDataset(data_file="dataset/validation.csv")
# test_dataset = ChnSentiCorpDataset(data_file="dataset/test.csv")
# print(f"train set size: {len(train_dataset)}")
# print(f"train set size: {len(val_dataset)}")
# print(f"train set size: {len(test_dataset)}")
# print(next(iter(train_dataset)))
# print(next(iter(val_dataset)))
# print(next(iter(test_dataset)))

# train_dataloader = DataLoader(
#     train_dataset, batch_size=4, shuffle=True, collate_fn=collote_fn
# )
# valid_dataloader = DataLoader(
#     val_dataset, batch_size=4, shuffle=False, collate_fn=collote_fn
# )
# test_dataloader = DataLoader(
#     val_dataset, batch_size=4, shuffle=False, collate_fn=collote_fn
# )

# batch = next(iter(train_dataloader))
# print(batch.keys())
# print(batch)
