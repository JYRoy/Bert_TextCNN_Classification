import random
import time
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW, get_scheduler
from sklearn.metrics import accuracy_score

from dataloader import *
from model import *

EPOCH_NUM = 10
BATCH_SIZE = 32
seed = 2024
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f"loss: {0:>7f}")
    finish_batch_num = (epoch - 1) * len(dataloader)

    model.train()
    for batch, batch_data in enumerate(dataloader, start=1):
        input_ids = batch_data[0].to(device)
        attention_mask = batch_data[1].to(device)
        token_type_ids = batch_data[2].to(device)
        labels = batch_data[3].to(device)

        logits = model(input_ids, attention_mask, token_type_ids)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(
            f"loss: {total_loss/(finish_batch_num + batch):>7f}"
        )
        progress_bar.update(1)
    return total_loss


def valid_loop(dataloader, model):
    progress_bar = tqdm(range(len(dataloader)))
    all_true, all_pred = [], []
    model.eval()
    for batch, batch_data in enumerate(dataloader, start=1):
        input_ids = batch_data[0].to(device)
        attention_mask = batch_data[1].to(device)
        token_type_ids = batch_data[2].to(device)
        labels = batch_data[3].to(device)

        logits = model(input_ids, attention_mask, token_type_ids)

        pred = torch.argmax(logits, -1)
        pred_labels = pred.cpu().numpy().tolist()
        true_labels = labels.cpu().numpy().tolist()
        all_pred.extend(pred_labels)
        all_true.extend(true_labels)
        progress_bar.update(1)
    acc = accuracy_score(all_true, all_pred)
    print("Validation Acc: ", acc)


train_dataset = ChnSentiCorpDataset(data_file="dataset/train.csv")
val_dataset = ChnSentiCorpDataset(data_file="dataset/validation.csv")
test_dataset = ChnSentiCorpDataset(data_file="dataset/test.csv")
print(f"train set size: {len(train_dataset)}")
print(f"train set size: {len(val_dataset)}")
print(f"train set size: {len(test_dataset)}")

train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_wo_bert
)
valid_dataloader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_wo_bert
)
test_dataloader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_wo_bert
)
model = BertTextCNNModel()
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.95), eps=1e-8)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=EPOCH_NUM * len(train_dataloader),
)


total_loss = 0.0
for t in range(EPOCH_NUM):
    print(f"Epoch {t+1}/{EPOCH_NUM}\n-------------------------------")
    total_loss = train_loop(
        train_dataloader, model, optimizer, lr_scheduler, t + 1, total_loss
    )
    valid_loop(valid_dataloader, model)

print("Done!")
