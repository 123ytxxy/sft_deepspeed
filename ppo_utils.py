import logging
import os
from dataclasses import dataclass
from typing import Dict, Sequence, Union, List
import datasets
import torch
from datasets import load_dataset, concatenate_datasets
import transformers
from transformers import AutoTokenizer
from torch.utils.data import DataLoader,RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

def build_reward_dataset(tokenizer, data_path, max_length, preprocessing_num_workers):
    """
     {"positive": prompt + answer_positive, "negative": prompt + answer_negative}, where the positive response is preferred.
    """
    def formatting_prompts_func(examples):
        chosens = []
        rejecteds = []
        for context, chosen, rejected in zip(examples['context'], examples['chosen'], examples['rejected']):
            prompt = ''
            for j in context:
                if j['role'] == 'human':
                    prompt += f"<|im_start|>user\n{j['text']}<|im_end|>\n".replace(' ', '')
                if j['role'] == 'assistant':
                    prompt += f"<|im_start|>assistant\n{j['text']}<|im_end|>\n"
            chosen = prompt + f"<|im_start|>assistant\n{chosen['text']}<|im_end|>".replace(' ', '')
            rejected = prompt + f"<|im_start|>assistant\n{rejected['text']}<|im_end|>".replace(' ', '')
            chosens.append(chosen)
            rejecteds.append(rejected)
        res = {}
        res['chosen']=chosens
        res['rejected'] = rejecteds
        return res
        
    def tokenization(examples):
        examples = formatting_prompts_func(examples)
        tokenized_chosen = tokenizer(examples['chosen'], return_attention_mask=False, add_special_tokens=False, truncation=True, max_length=max_length)
        tokenized_rejected = tokenizer(examples['rejected'], return_attention_mask=False, add_special_tokens=False, truncation=True, max_length=max_length)
        print(tokenized_chosen)
        # chosen_input_ids = []
        # chosen_attention_mask = []
        # rejected_input_ids = []
        # rejected_attention_mask = []
        # for chosen, rejected in zip(tokenized_chosen['input_ids'], tokenized_rejected['input_ids']):
            # chosen_input_ids.append(chosen)
            # chosen_attention_mask.append(chosen['attention_mask'])
            # rejected_input_ids.append(rejected)
            # rejected_attention_mask.append(rejected['attention_mask'])
        results = {'chosen_input_ids':tokenized_chosen['input_ids'], 'rejected_input_ids':tokenized_rejected['input_ids']}
        return results
    all_datasets = []
    if not isinstance(data_path,(list,tuple)):
        file_path = [data_path]
    for data_path in file_path:

        raw_dataset = load_dataset('json', data_files=data_path, split='train')
        tokenization_func = tokenization
        
        tokenized_dataset = raw_dataset.map(
                    tokenization_func,
                    batched=True,  # 批量处理数据
                    num_proc=preprocessing_num_workers,
                    remove_columns=['context', 'chosen', 'rejected'],
                    keep_in_memory=False,  # 不将数据集保存在整个内存中
                    desc="preprocessing on dataset",
                )
        tokenized_dataset.set_format('torch')
        all_datasets.append(tokenized_dataset)
    all_datasets = concatenate_datasets(all_datasets)
    return all_datasets

