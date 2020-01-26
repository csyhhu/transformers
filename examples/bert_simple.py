"""
A test of bert in evaluation
"""
import argparse
import logging
from tqdm import tqdm
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch

from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
}

from transformers.data.processors.glue import \
    glue_convert_examples_to_features as convert_examples_to_features
from transformers.data.metrics import glue_compute_metrics as compute_metrics
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

# parser = argparse.ArgumentParser()
# args = parser.parse_args()

# -----------------------------------
task_name = 'mrpc'
data_dir = './glue_data/MRPC'
model_type = 'bert'
cache_dir = './Results/%s-cache' %task_name
config_name = ""
tokenizer_name = ""
model_name_or_path = "bert-base-uncased"
max_seq_length = 128
eval_batch_size = 64
use_cuda = torch.cuda.is_available()
device = 'cuda'
# -----------------------------------

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.warning("This is a test")

processor = processors[task_name]()
output_mode = output_modes[task_name]
label_list = processor.get_labels() # ['0', '1']
num_labels = len(label_list)
# ds

config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
config = config_class.from_pretrained(
    config_name if config_name else model_name_or_path,
    num_labels=num_labels,
    finetuning_task=task_name,
    cache_dir=cache_dir,
)

tokenizer = tokenizer_class.from_pretrained(
    tokenizer_name if tokenizer_name else model_name_or_path,
    do_lower_case=True,
    cache_dir=cache_dir,
)

model = model_class.from_pretrained(
    model_name_or_path,
    from_tf=bool(".ckpt" in model_name_or_path),
    config=config,
    cache_dir=cache_dir if cache_dir else None,
)

if use_cuda:
    model.cuda()

# Evaluate
# def load_and_cache_examples(task_name, tokenizer, evaluate=False):
processor = processors[task_name]()
output_mode = output_modes[task_name]
label_list = processor.get_labels()
examples = (processor.get_dev_examples(data_dir))
features = convert_examples_to_features(
    examples,
    tokenizer,
    label_list=label_list,
    max_length=max_seq_length,
    output_mode=output_mode,
    pad_on_left=bool(model_type in ["xlnet"]),  # pad on the left for xlnet
    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
    pad_token_segment_id=4 if model_type in ["xlnet"] else 0,
)
all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
if output_mode == "classification":
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
elif output_mode == "regression":
    all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

eval_dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
eval_dataloader = DataLoader(eval_dataset, batch_size=eval_batch_size)

# Eval!
logger.info("***** Running evaluation *****")
logger.info("  Num examples = %d", len(eval_dataset))
logger.info("  Batch size = %d", eval_batch_size)
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
                batch[2] if model_type in ["bert", "xlnet", "albert"] else None
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