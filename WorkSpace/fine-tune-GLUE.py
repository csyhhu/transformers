"""
This code simplifies run_glue.py to perform fine-tune on BERT using GLUE data
"""

import argparse
import logging
from tqdm import tqdm
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
from torch.utils.data.dataloader import DataLoader

from transformers.data.processors import glue_processors, glue_output_modes
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from utils.dataset import load_and_cache_examples
from utils.miscellaneous import MODEL_CLASSES

# -----------------------------------
task_name = 'mrpc'
data_dir = '../glue_data/MRPC'
model_type = 'bert'
cache_dir = '../Results/%s-cache' %task_name
config_name = ""
tokenizer_name = ""
model_name_or_path = "bert-base-uncased"
max_seq_length = 128
train_batch_size = 128
eval_batch_size = 64
weight_decay = 5e-4
learning_rate = 1e-3
adam_epsilon = 1e-8
warmup_steps = 10
t_total = 1e3
use_cuda = torch.cuda.is_available()
device = 'cuda'
# -----------------------------------

processor = glue_processors[task_name]()
output_mode = glue_output_modes[task_name]
label_list = processor.get_labels() # ['0', '1']
num_labels = len(label_list)

# ------------------
# Get configuration
# ------------------
config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
config = config_class.from_pretrained(
    config_name if config_name else model_name_or_path,
    num_labels=num_labels,
    finetuning_task=task_name,
    cache_dir=cache_dir,
)

# --------------
# Get Tokenizer
# --------------
tokenizer = tokenizer_class.from_pretrained(
    tokenizer_name if tokenizer_name else model_name_or_path,
    do_lower_case=True,
    cache_dir=cache_dir,
)

# ---------
# Get Model
# ---------
model = model_class.from_pretrained(
    model_name_or_path,
    from_tf=bool(".ckpt" in model_name_or_path),
    config=config,
    cache_dir=cache_dir if cache_dir else None,
)

# ------------
# Load dataset
# -------------
train_dataset = load_and_cache_examples(
    dataset_name='glue', task_name=task_name, tokenizer=tokenizer, evaluate=False)
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size)

# --------------------
# Initialize Optimizer
# --------------------
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters()
                   if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
    },
    {"params": [p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
)

# --------------
# Begin Training
# --------------
n_epoch = 1
tr_loss = 0
global_step = 0
# for epoch in range(n_epoch):
epoch_iterator = tqdm(train_dataloader, desc="Iteration")
for step, batch in enumerate(epoch_iterator):
    model.train()
    batch = tuple(t.to(device) for t in batch)
    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
    outputs = model(**inputs)
    losses = outputs[0]

    losses.backward()
    optimizer.step()
    scheduler.step()  # Update learning rate schedule
    model.zero_grad()
    global_step += 1

    tr_loss += losses.item()