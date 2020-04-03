"""
This code evaluates bert
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
from transformers.data.metrics import glue_compute_metrics as compute_metrics

from utils.dataset import load_and_cache_examples
from utils.miscellaneous import MODEL_CLASSES

import argparse
parser = argparse.ArgumentParser(description='BERT evaulation')
parser.add_argument('--model_type', '-m', type=str,
                    default='qbert', help='Model Arch')
args = parser.parse_args()
print(args)
# -----------------------------------
task_name = 'mrpc'
data_dir = '../glue_data/MRPC'
model_type = args.model_type
cache_dir = './Results/BERT-GLUE-%s/raw' %(task_name.upper())
# pretrain_dir = './Results/BERT-GLUE-%s/pretrain' %(task_name.upper())
pretrain_dir = None
model_name = "bert-base-uncased"
max_seq_length = 128
train_batch_size = 128
eval_batch_size = 32
weight_decay = 5e-4
learning_rate = 1e-3
adam_epsilon = 1e-8
warmup_steps = 10
t_total = 1e3
use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
# print(device)
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
    # config_name if config_name else model_name_or_path,
    # num_labels=num_labels,
    # finetuning_task=task_name,
    # cache_dir=cache_dir,
    pretrained_model_name_or_path = model_name if pretrain_dir is None else pretrain_dir,
    cache_dir=cache_dir, do_lower_case=True,
)

# Add bitW
config.bitW = 1

# --------------
# Get Tokenizer
# --------------
tokenizer = tokenizer_class.from_pretrained(
    # tokenizer_name if tokenizer_name else model_name_or_path,
    #
    # cache_dir=cache_dir,
    pretrained_model_name_or_path=model_name if pretrain_dir is None else pretrain_dir,
    cache_dir=cache_dir, do_lower_case=True,
)

# ---------
# Get Model
# ---------
model = model_class.from_pretrained(
    # model_name_or_path,
    # # from_tf=bool(".ckpt" in model_name_or_path),
    # config=config,
    # cache_dir=cache_dir if cache_dir else None,
    pretrained_model_name_or_path=model_name if pretrain_dir is None else pretrain_dir,
    cache_dir=cache_dir
)

model.to(device)

# ------------
# Load dataset
# -------------
eval_dataset = load_and_cache_examples(
    dataset_name='glue', task_name=task_name, tokenizer=tokenizer, evaluate=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=eval_batch_size)

eval_loss = 0.0
nb_eval_steps = 0
preds = None
out_label_ids = None
results = {}
model.eval()

for batch in tqdm(eval_dataloader, desc="Evaluating"):
    batch = tuple(t.to(device) for t in batch)

    with torch.no_grad():
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
        if model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch[2] if model_type in ["bert", "qbert", "xlnet", "albert"] else None
            )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
        outputs = model(**inputs)
        tmp_eval_loss, logits = outputs[:2]

        eval_loss += tmp_eval_loss.mean().item()
    nb_eval_steps += 1
    if preds is None:
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()
    else:
        preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
        out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

eval_loss = eval_loss / nb_eval_steps
if output_mode == "classification":
    preds = np.argmax(preds, axis=1)
elif output_mode == "regression":
    preds = np.squeeze(preds)
result = compute_metrics(task_name, preds, out_label_ids)
results.update(result)

print(results)