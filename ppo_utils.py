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
import torch
from torch import nn

def build_reward_dataset(tokenizer, data_path, max_seq_length, preprocessing_num_workers):
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
        tokenized_chosen = tokenizer(examples['chosen'], return_attention_mask=False, add_special_tokens=False, truncation=True, padding="max_length", max_length=max_seq_length)
        tokenized_rejected = tokenizer(examples['rejected'], return_attention_mask=False, add_special_tokens=False, truncation=True, padding="max_length", max_length=max_seq_length)
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
                    batched=True, 
                    num_proc=preprocessing_num_workers,
                    remove_columns=['context', 'chosen', 'rejected'],
                    keep_in_memory=False,  
                    desc="preprocessing on dataset",
                )
        tokenized_dataset.set_format('torch')
        all_datasets.append(tokenized_dataset)
    all_datasets = concatenate_datasets(all_datasets)
    return all_datasets

@dataclass
class DataCollatorForRewardModelDataset(object):
    """Collate examples for rewardmodel training."""
    tokenizer: transformers.PreTrainedTokenizer
    # __call__ 方法使得该类的实例可以像函数一样被调用
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        chosen_input_ids, rejected_input_ids = tuple([instance[key] for instance in instances] for key in ('chosen_input_ids', 'rejected_input_ids'))
        # chosen_input_ids = torch.nn.utils.rnn.pad_sequence(
        #     chosen_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        # )  
        # rejected_input_ids = torch.nn.utils.rnn.pad_sequence(
        #     rejected_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        
        return dict(
            chosen_input_ids=chosen_input_ids,
            rejected_input_ids=rejected_input_ids,
            chosen_attention_mask=chosen_input_ids.ne(self.tokenizer.pad_token_id),
            rejected_attention_mask=rejected_input_ids.ne(self.tokenizer.pad_token_id)
        ) 
    

## Note that the following code is modified from
## https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/reward_model.py
class RewardModel(nn.Module):

    def __init__(self,
                 base_model,
                 tokenizer,
                 num_padding_at_beginning=0,
                 compute_fp32_loss=False):
        super().__init__()
        self.config = base_model.config
        self.num_padding_at_beginning = num_padding_at_beginning
        if hasattr(self.config, "word_embed_proj_dim"):
            # `OPT` models use word_embed_proj_dim as final output
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L497
            self.v_head = nn.Linear(self.config.word_embed_proj_dim,
                                    1,
                                    bias=False)
        else:
            # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
            self.config.n_embd = self.config.hidden_size if hasattr(
                self.config, "hidden_size") else self.config.n_embd
            self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.rwtransformer = base_model
        # base_model： AutoModel 会返回last_hidden_state
        self.PAD_ID = tokenizer.pad_token_id
        self.compute_fp32_loss = compute_fp32_loss

    def gradient_checkpointing_enable(self):
        self.rwtransformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.rwtransformer.gradient_checkpointing_disable()

    def forward(self,
                chosen_input_ids=None,
                rejected_input_ids=None,
                chosen_attention_mask=None,
                rejected_attention_mask=None,
                past_key_values=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                use_cache=False):
        loss = None

        if self.config.model_type == "llama":
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)

        chosen_transformer_outputs = self.rwtransformer(
            chosen_input_ids,
            past_key_values=past_key_values,
            attention_mask=chosen_attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs)
        rejected_transformer_outputs = self.rwtransformer(
            rejected_input_ids,
            past_key_values=past_key_values,
            attention_mask=rejected_attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs)

        chosen_hidden_states = chosen_transformer_outputs[0]
        chosen_rewards = self.v_head(chosen_hidden_states).squeeze(-1)
        rejected_hidden_states = rejected_transformer_outputs[0]
        rejected_rewards = self.v_head(rejected_hidden_states).squeeze(-1)

        chosen_mean_scores = []
        rejected_mean_scores = []

        assert len(chosen_input_ids.shape) == 2  
        bs = chosen_input_ids.shape[0]
        seq_len = chosen_input_ids.shape[1]


        # Compute pairwise loss. Only backprop on the different tokens before padding
        loss = 0.
        for i in range(bs):
            chosen_id = chosen_input_ids[i]
            rejected_id = rejected_input_ids[i]
            chosen_reward = chosen_rewards[i]
            rejected_reward = rejected_rewards[i]

            c_inds = (chosen_id == self.PAD_ID).nonzero()  # .nonzero() 返回非0元素的索引
            # 保存填充标记的位置，用于确定序列的实际长度
            c_ind = c_inds[self.num_padding_at_beginning].item() if len(
                c_inds
            ) > self.num_padding_at_beginning else seq_len  # OPT model pads the first token, so we need to use the second padding token as the end of the sequence
            # 初始填充标记的位置
            # 确定chosen_id的有效结束位置
            check_divergence = (chosen_id != rejected_id).nonzero()
            # 返回所有不同的位置索引

            if len(check_divergence) == 0:
                end_ind = rejected_reward.size(-1)
                divergence_ind = end_ind - 1  # 最后一个位置的索引
                r_ind = c_ind  # 因为两个样本相同
            else:
                # Check if there is any padding otherwise take length of sequence
                r_inds = (rejected_id == self.PAD_ID).nonzero()  # 标记出填充位置
                r_ind = r_inds[self.num_padding_at_beginning].item(
                ) if len(r_inds) > self.num_padding_at_beginning else seq_len
                # 找到 rejected_id 中的有效结束位置
                end_ind = max(c_ind, r_ind)  # 最大值，确保覆盖两个序列的有效部分
                divergence_ind = check_divergence[0]  # 取第一个分歧点的位置
            assert divergence_ind > 0
            # 如果两个样本相同，就是取最后一个位置的奖励
            # 不相同，从第一个不相同位置到最后
            c_truncated_reward = chosen_reward[divergence_ind:end_ind]
            r_truncated_reward = rejected_reward[divergence_ind:end_ind]
            chosen_mean_scores.append(
                chosen_reward[c_ind - 1])  #use the end score for reference
            rejected_mean_scores.append(rejected_reward[r_ind - 1])

            if self.compute_fp32_loss:
                c_truncated_reward = c_truncated_reward.float()
                r_truncated_reward = r_truncated_reward.float()
            loss += -torch.nn.functional.logsigmoid(c_truncated_reward -
                                                    r_truncated_reward).mean()

        loss = loss / bs
        chosen_mean_scores = torch.stack(chosen_mean_scores)
        rejected_mean_scores = torch.stack(rejected_mean_scores)
        return {
            "loss": loss,
            "chosen_mean_scores": chosen_mean_scores,
            "rejected_mean_scores": rejected_mean_scores,
        }