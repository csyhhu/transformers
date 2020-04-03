"""
This code performs compressing fine-tuning
"""
import argparse
import logging
from tqdm import tqdm
import numpy as np
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
from torch.utils.data.dataloader import DataLoader

from transformers.data.processors import glue_processors, glue_output_modes
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from utils.dataset import load_and_cache_examples
from utils.miscellaneous import MODEL_CLASSES, progress_bar
from utils.train import evaluate

parser = argparse.ArgumentParser()
# -------
# Hyper Parameters for Training
# -------
parser.add_argument("--train_batch_size", '-bs', default=32, type=int,
                    help=" ")
parser.add_argument("--eval_batch_size", '-eval_bs', default=16, type=int,
                    help=" ")
parser.add_argument("--learning_rate", '-lr', default=2e-5, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", '-wd', default=0.0, type=float,
                    help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument("--num_train_epochs", "-epoch", default=3, type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_steps", default=0, type=int,
                    help="Linear warmup over warmup_steps.")
parser.add_argument("--first_eval", action="store_true",
                    help="Whether to perform first evaluation",)
parser.add_argument( "--prune", action="store_true",
                    help="Whether to perform pruning",)
parser.add_argument( "--bitW", '-bitW', default=8, type=int,
                    help="")
parser.add_argument( "--CR", '-CR', default=0.3, type=float,
                    help="")
# --------
# Mixed-Precision Training
# ---------
# parser.add_argument(
#     "--fp16",
#     action="store_true",
#     help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
# )
# parser.add_argument(
#     "--fp16_opt_level",
#     type=str,
#     default="O1",
#     help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
#     "See details at https://nvidia.github.io/apex/amp.html",
# )
args = parser.parse_args()
print(args)
# -----------------------------------
task_name = 'mrpc'
data_dir = '../glue_data/MRPC'
model_type = 'sbert' if args.prune else 'qbert'
# cache_dir = '../Results/%s-cache' %task_name
cache_dir = './cache'
config_name = ""
tokenizer_name = ""
model_name_or_path = "bert-base-uncased"
max_seq_length = 128
train_batch_size = args.train_batch_size
eval_batch_size = args.eval_batch_size
weight_decay = args.weight_decay
learning_rate = args.learning_rate
adam_epsilon = args.adam_epsilon
warmup_steps = args.warmup_steps
max_grad_norm = args.max_grad_norm
n_epoch = args.num_train_epochs
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
if args.prune:
    config.CR = args.CR
else:
    config.bitW = args.bitW

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
    # from_tf=bool(".ckpt" in model_name_or_path),
    config=config,
    cache_dir=cache_dir if cache_dir else None,
)
print('Model Loaded Successfully')
model.to(device)

# ------------
# Load dataset
# -------------
train_dataset = load_and_cache_examples(
    dataset_name='glue', task_name=task_name, tokenizer=tokenizer, evaluate=False
)
eval_dataset = load_and_cache_examples(
    dataset_name='glue', task_name=task_name, tokenizer=tokenizer, evaluate=True
)
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size)
eval_dataloader = DataLoader(eval_dataset, batch_size=eval_batch_size)

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
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=len(train_dataloader) * n_epoch
)

if args.first_eval:
    result = evaluate(task_name, model, eval_dataloader, model_type) # ['acc', 'f1', 'acc_f1']
    print(result)

# --------------
# Begin Training
# --------------
for epoch_idx in range(n_epoch):

    print('Epoch: %d' %epoch_idx)
    model.train()
    train_loss = 0
    # global_step = 0
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
        if model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch[2] if model_type in ["bert", "xlnet", "albert"] else None
            )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
        outputs = model(**inputs)
        losses = outputs[0]

        # model.zero_grad()
        losses.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        model.zero_grad()
        # global_step += 1

        # ------
        # Record
        # ------
        # result = compute_metrics(task_name, preds, out_label_ids)
        train_loss += losses.item()

        progress_bar(step, len(train_dataloader), "Loss: %.3f" % (train_loss / (step + 1)))

    result = evaluate(task_name, model, eval_dataloader, model_type)  # ['acc', 'f1', 'acc_f1']
    print(result)
