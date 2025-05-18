import logging
import os
from dataclasses import dataclass
from typing import Dict, Sequence, Union, List
import datasets
import torch
from datasets import load_dataset, concatenate_datasets
import transformers
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader,RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch
from torch import nn
import deepspeed

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

# This function is a modified version of code available in the from_pretrained API of HuggingFace Transformers
# The code is copied and modified from: https://github.com/huggingface/transformers/blob/5ee9693a1c77c617ebc43ef20194b6d3b674318e/src/transformers/modeling_utils.py#L498
# This function helps load a HF format checkpoint into a DeepSpeed wrapped model that has been sharded using ZeRO Stage 3
def load_state_dict_into_model(model_to_load=None,
                               state_dict=None,
                               start_prefix="",
                               zero_stage=0):

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    error_msgs = []

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module: nn.Module, state_dict, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        # Parameters of module and children will start with prefix. We can exit early if there are none in this
        # state_dict
        if len([key for key in state_dict if key.startswith(prefix)]) > 0:
            if zero_stage == 3:
                # In sharded models, each shard has only part of the full state_dict, so only gather
                # parameters that are in the current state_dict.
                named_parameters = dict(
                    module.named_parameters(prefix=prefix[:-1], recurse=False))
                params_to_gather = [
                    named_parameters[k] for k in state_dict.keys()
                    if k in named_parameters
                ]
                if len(params_to_gather) > 0:
                    # because zero3 puts placeholders in model params, this context
                    # manager gathers (unpartitions) the params of the current layer, then loads from
                    # the state dict and then re-partitions them again
                    with deepspeed.zero.GatheredParameters(params_to_gather,
                                                           modifier_rank=0):
                        if torch.distributed.get_rank() == 0:
                            module._load_from_state_dict(*args)
            else:
                module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, state_dict, prefix + name + ".")

    load(model_to_load, state_dict, prefix=start_prefix)
    # Delete `state_dict` so it could be collected by GC earlier. Note that `state_dict` is a copy of the argument, so
    # it's safe to delete it.
    del state_dict

    return error_msgs

def create_critic_model(model_path,
                        tokenizer,
                        ds_config,
                        num_padding_at_beginning=0,
                        rlhf_training=False,
                        dropout=None,
                        zero_stage=0,
                        compute_fp32_loss=False):
    # OPT model family always put a padding token at the beginning of the sequence,
    # we did not find this in other models but not sure if it is a general rule

    critic_model = AutoModel.from_pretrained(model_path)

    critic_model = RewardModel(
        critic_model,
        tokenizer,
        num_padding_at_beginning=num_padding_at_beginning,
        compute_fp32_loss=compute_fp32_loss)

    if rlhf_training:
        # load critic model from checkpoint

        model_ckpt_path = os.path.join(model_path, 'pytorch_model.bin')
        assert os.path.exists(
            model_ckpt_path
        ), f"Cannot find model checkpoint at {model_ckpt_path}"

        model_ckpt_state_dict = torch.load(model_ckpt_path, map_location='cpu')
    
        # load critic model from checkpoint with zero-stage 3 compatibility
        # this functionality may be moved to DS checkpoint load API in future
        load_state_dict_into_model(critic_model,
                                   model_ckpt_state_dict,
                                   "",
                                   zero_stage=zero_stage)

    return critic_model