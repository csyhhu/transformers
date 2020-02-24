"""
Utilis function for loading dataset
"""
import os

import torch
from torch.utils.data.dataset import TensorDataset
from transformers.data.processors import \
    glue_processors, glue_output_modes, glue_convert_examples_to_features

def load_and_cache_examples(dataset_name, task_name, tokenizer,
                            max_seq_length = 128, model_type='bert',
                            evaluate=False):

    data_dir_list = {
        'glue': ['../glue_data/%s' %task_name.upper()]
    }

    data_dir = None
    for data_dir in data_dir_list[dataset_name]:
        if os.path.exists(data_dir):
            print('Found %s data in %s' %(dataset_name, data_dir))
            break
    if data_dir is None:
        raise ValueError('%s data not found' %(dataset_name))

    if dataset_name == 'glue':
        processor = glue_processors[task_name]()
        output_mode = glue_output_modes[task_name]
        label_list = processor.get_labels()
        examples = (
            processor.get_dev_examples(data_dir) if evaluate else
            processor.get_train_examples(data_dir)
        )

        features = glue_convert_examples_to_features(
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

        dataset = TensorDataset(all_input_ids, all_attention_mask,
                                all_token_type_ids, all_labels)
    else:
        raise NotImplementedError

    return dataset

if __name__ == '__main__':

    from transformers import BertTokenizer, BertConfig
    from torch.utils.data.dataloader import DataLoader
    from utils.miscellaneous import MODEL_CLASSES

    eval_batch_size = 128
    task_name = 'mrpc'
    model_type = 'bert'
    # config_name = ""
    # tokenizer_name = ""
    pretrain_dir = './Results/BERT-GLUE-MRPC/pretrain'

    processor = glue_processors[task_name]()
    output_mode = glue_output_modes[task_name]
    label_list = processor.get_labels()  # ['0', '1']
    num_labels = len(label_list)

    # ------------------
    # Get configuration
    # ------------------
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    # config = config_class.from_pretrained(
    #     config_name if config_name else model_name_or_path,
    #     num_labels=num_labels,
    #     finetuning_task=task_name,
    #     cache_dir=cache_dir,
    # )

    # --------------
    # Get Tokenizer
    # --------------
    tokenizer = tokenizer_class.from_pretrained(
        pretrained_model_name_or_path=pretrain_dir
    )

    eval_dataset = load_and_cache_examples(
        dataset_name='glue', task_name='mrpc', tokenizer=tokenizer, evaluate=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=eval_batch_size)