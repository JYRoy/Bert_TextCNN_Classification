import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import tiktoken
import numpy as np

import torch
from torch.utils.data import Dataset, random_split, DataLoader
from transformers import BertTokenizer, BertModel

max_input_length = 1024
max_target_length = 1024

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
pretrained_model = BertModel.from_pretrained("bert-base-chinese")
pretrained_model = pretrained_model.to("cuda")

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


def collate_fn_wo_bert(batch_samples):
    sents = [i["text"] for i in batch_samples]
    labels = [i["label"] for i in batch_samples]

    data = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        truncation=True,
        padding="max_length",
        max_length=500,
        return_tensors="pt",
        return_length=True,
    )

    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    labels = torch.LongTensor(labels)
    return input_ids, attention_mask, token_type_ids, labels


def collate_fn_with_bert(batch_samples):
    sents = [i["text"] for i in batch_samples]
    labels = [i["label"] for i in batch_samples]

    data = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        truncation=True,
        padding="max_length",
        max_length=500,
        return_tensors="pt",
        return_length=True,
    )

    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    labels = torch.LongTensor(labels)
    with torch.no_grad():
        out = pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
    return out, labels


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
#     train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn_wo_bert
# )
# valid_dataloader = DataLoader(
#     val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn_wo_bert
# )
# test_dataloader = DataLoader(
#     val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn_wo_bert
# )

# batch = next(iter(train_dataloader))
# print(batch.keys())
# print(batch)

# train_dataloader = DataLoader(
#     train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn_wo_bert
# )
# valid_dataloader = DataLoader(
#     val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn_wo_bert
# )
# test_dataloader = DataLoader(
#     val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn_wo_bert
# )

# batch = next(iter(train_dataloader))
# print(batch)
