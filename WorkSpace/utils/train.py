import os
import numpy as np

import torch
from transformers import glue_compute_metrics

from utils.miscellaneous import progress_bar

def evaluate(task_name, model, eval_dataloader, model_type, output_mode = 'classification', device='cuda'):
    # results = {}

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch_idx, batch in enumerate(eval_dataloader):
        model.eval()
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

        progress_bar(batch_idx, len(eval_dataloader), 'Evaluating...')

    eval_loss = eval_loss / nb_eval_steps
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)

    result = glue_compute_metrics(task_name, preds, out_label_ids) # [
    # print(result)
    # results.update(result)
    return result