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

IGNORE_INDEX = -100

logger = logging.getLogger('__name__')

PROMPT_TEMPLATE = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    )

def build_instruction_dataset(data_path: Union[List[str],str],
                tokenizer: transformers.PreTrainedTokenizer,
                max_seq_length: int, data_cache_dir = None,
                preprocessing_num_workers = None,
                ):

    def tokenization(examples):
        sources = []
        targets = []
        prompt = PROMPT_TEMPLATE
        for instruction, input, output in zip(examples['instruction'],examples['input'],examples['output']):
            if input is not None and input !="":
                instruction = instruction+'\n'+input
            source = prompt.format_map({'instruction':instruction})
            target = f"{output}{tokenizer.eos_token}"

            sources.append(source)
            targets.append(target)

        tokenized_sources = tokenizer(sources,return_attention_mask=False)  # 会在开头加上special_tokens 128000
        tokenized_targets = tokenizer(targets,return_attention_mask=False,add_special_tokens=False)  # 需要在结尾加上128001，所以不在这里使用add_special_tokens，否则会在开头加

        all_input_ids = []
        all_labels = []
        for s,t in zip(tokenized_sources['input_ids'],tokenized_targets['input_ids']):
            input_ids = torch.LongTensor(s + t)[:max_seq_length]
            labels = torch.LongTensor([IGNORE_INDEX] * len(s) + t)[:max_seq_length]
            assert len(input_ids) == len(labels)
            all_input_ids.append(input_ids)
            all_labels.append(labels)

        results = {'input_ids':all_input_ids, 'labels': all_labels}
        return results


    logging.warning("building dataset...")
    all_datasets = []

    if not isinstance(data_path,(list,tuple)):
        data_path = [data_path]
    for file in data_path:

        if data_cache_dir is None:
            data_cache_dir = str(os.path.dirname(file))
        cache_path = os.path.join(data_cache_dir,os.path.basename(file).split('.')[0])
        os.makedirs(cache_path, exist_ok=True)
        try:
            processed_dataset = datasets.load_from_disk(cache_path)
            logger.info(f'training datasets-{file} has been loaded from disk')
        except Exception:
            raw_dataset = load_dataset("json", data_files=file, cache_dir=cache_path)  # data_files加载路径，cache_dir数据缓存路径
            tokenization_func = tokenization
            tokenized_dataset = raw_dataset.map(
                tokenization_func,
                batched=True,  # 批量处理数据
                num_proc=preprocessing_num_workers,
                remove_columns=["instruction","input","output"],
                keep_in_memory=False,  # 不将数据集保存在整个内存中
                desc="preprocessing on dataset",
            )
            processed_dataset = tokenized_dataset
            processed_dataset.save_to_disk(cache_path)
        processed_dataset.set_format('torch')
        all_datasets.append(processed_dataset['train'])
    all_datasets = concatenate_datasets(all_datasets)
    return all_datasets

# 装饰器，简化类的创建过程
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    # __call__ 方法使得该类的实例可以像函数一样被调用
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )  # 填充到批次中的最长长度
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)  # 损失计算时会忽略这个值
        # 计算损失的时候loss_fct = CrossEntropyLoss() ignore_index默认为-100
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )   # .ne not equal 的缩写

if __name__ == '__main__':
    IGNORE_INDEX = -100
    DEFAULT_PAD_TOKEN = "[PAD]"
    tokenizer = AutoTokenizer.from_pretrained('../llama_1B')
    # 这需要重新训练一个tokenizer，否则设为eos_token比较好
    if tokenizer.pad_token is None: 
        tokenizer.add_special_tokens(dict(pad_token=DEFAULT_PAD_TOKEN))
        tokenizer.padding_side = 'right'
    # print(len(tokenizer))  加在了结尾，加入后为128257 id为128256
    #  微调数据，在label中包括输入提示和需要生成的答案，但是输入的提示部分不计算损失，所以在label中的对应位置为-100
    train_dataset = build_instruction_dataset(data_path='./alpaca_data_zh_51k.json', tokenizer=tokenizer,
                max_seq_length=512, data_cache_dir = None,
                preprocessing_num_workers = 1,
                )
    # model.resize_token_embeddings(len(tokenizer))
    # 抽样器
    # print(tokenizer.pad_token_id, tokenizer.pad_token, tokenizer.all_special_tokens, tokenizer.all_special_ids)
    train_sampler =RandomSampler(data_source=train_dataset, replacement=False,generator=torch.Generator().manual_seed(42))
    collate_fn = DataCollatorForSupervisedDataset(tokenizer)
    train_dataloader=DataLoader(train_dataset,
                        collate_fn=collate_fn,
                        sampler=train_sampler,
                        batch_size=16)


    
    