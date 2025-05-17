from datasets import load_dataset, DatasetDict, Dataset
import numpy as np
from transformers import AutoTokenizer
# import torch.distributed as dist
import deepspeed.comm as dist
from torch.utils.data import DataLoader, DistributedSampler
import os
import torch
import random
from safetensors.torch import save_file as safe_save_file
from deepspeed.accelerator import get_accelerator
import matplotlib.pyplot as plt

GLOBAL_BATCH_SIZE = 64
MICRO_BATCH_SIZE = 32

def create_pretrain_dataset(local_rank, data_path, data_split, output_path, 
                            seed, tokenizer, max_seq_len):
    
    np.random.seed(seed + local_rank)

    # 加载原始数据
    raw_data = load_dataset('text', data_files=data_path, split='train')

    # 分割数据集
    if data_split != "1":
        splits = list(map(float, data_split.split(',')))  
        split_dataset = raw_data.train_test_split(test_size=splits[0], seed=seed)
    else:
        split_dataset = DatasetDict({"train": raw_data})
    
    # 分布式分片
    if dist.is_initialized():
        train_dataset = split_dataset["train"].shard(
            num_shards=dist.get_world_size(),
            index=local_rank
        )
        eval_dataset = (
            split_dataset["test"].shard(
                num_shards=dist.get_world_size(),
                index=local_rank
            ) 
            if "test" in split_dataset 
            else None
        )
    else:
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"] if "test" in split_dataset else None

    # train_dataset = split_dataset["train"]
    # eval_dataset = split_dataset["test"] if "test" in split_dataset else None

    # 预处理函数
    def preprocess(examples):
        batch = tokenizer(
            examples["text"],
            max_length=max_seq_len,
            truncation=True,
            padding='max_length',
            add_special_tokens=True
        )
        # 处理 labels 中的 pad_token
        input_ids = batch["input_ids"]
        labels = input_ids.copy()
        # for seq in input_ids:
        #     labels.append([-100 if token == tokenizer.pad_token_id else token for token in seq])
        batch["labels"] = labels
        return batch

    # 预处理数据集
    train_dataset = train_dataset.map(
        preprocess,
        batched=True,
        remove_columns=["text"],
        num_proc=1
    )

    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            preprocess,
            batched=True,
            remove_columns=["text"],
            num_proc=1
        )

    # # 保存数据集（仅 rank 0）
    # if dist.get_rank() == 0:
    #     os.makedirs(output_path, exist_ok=True)
    #     train_dataset.save_to_disk(f"{output_path}/processed_train")
    #     if eval_dataset is not None:
    #         eval_dataset.save_to_disk(f"{output_path}/processed_eval")

    # dist.barrier()

    # # 加载数据集
    # train_dataset = Dataset.load_from_disk(f"{output_path}/processed_train")
    # if eval_dataset is not None:
    #     eval_dataset = Dataset.load_from_disk(f"{output_path}/processed_eval")

    return train_dataset, eval_dataset

# 一组使用权重衰减，另一组则不使用
# 这种参数分组有助于正则化模型，防止过拟合，并允许对特定参数应用不同的学习设置
def get_optimizer_grouped_parameters(model,
                                    weight_decay,
                                    no_decay_name_list=[
                                    "bias", "LayerNorm.weight", "layernorm.weight", "norm.weight"
                                    ]):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list) and p.requires_grad)
                ],
            "weight_decay": weight_decay
            
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n for nd in no_decay_name_list) and p.requires_grad)
                    ],
           "weight_decay": 0.0
          
        },
        ]
    return optimizer_grouped_parameters

def get_train_ds_config(
                        offload,
                        stage=2,
                        enable_hybrid_engine=False,
                        inference_tp_size=1,
                        release_inference_cache=False,
                        pin_parameters=True,
                        tp_gather_partition_size=8,
                        max_out_tokens=512,
                        enable_tensorboard=False,
                        tb_path="./",
                        tb_name=""):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
    "stage": stage,
    "offload_optimizer": {
    "device": device
    },
    "stage3_param_persistence_threshold": 1e4,
    "stage3_max_live_parameters": 3e7,
    "stage3_prefetch_bucket_size": 3e7,
    "memory_efficient_linear": False
    }
    return {
            "train_batch_size": GLOBAL_BATCH_SIZE,
            "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
            "steps_per_print": 10,
            "zero_optimization": zero_opt_dict,
            "fp16": {
            "enabled": True,
            "loss_scale_window": 100
            },
            "gradient_clipping": 1.0,
            "prescale_gradients": False,
            "wall_clock_breakdown": False,
            "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
            },
            "tensorboard": {
            "enabled": enable_tensorboard,
            "output_path": f"{tb_path}/ds_tensorboard_logs/",
            "job_name": f"{tb_name}_tensorboard"
            }
            }

def get_eval_ds_config(offload, stage=0):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
    "stage": stage,
    "stage3_param_persistence_threshold": 1e4,
    "offload_optimizer": {
    "device": device
    },
    "memory_efficient_linear": False
    }
    return {
            "train_batch_size": GLOBAL_BATCH_SIZE,
            "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
            "steps_per_print": 10,
            "zero_optimization": zero_opt_dict,
            "fp16": {
            "enabled": True
            },
            "gradient_clipping": 1.0,
            "prescale_gradients": False,
            "wall_clock_breakdown": False
            }

def set_random_seed(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        get_accelerator().manual_seed_all(seed)


def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)

def to_device(batch, device):
    if isinstance(batch, dict):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    elif isinstance(batch, list):
        return [v.to(device) if isinstance(v, torch.Tensor) else v for v in batch]
    else:
        raise TypeError("Batch data type not supported.")

def save_hf_format(model, tokenizer, output_dir):
 # used to save huggingface format, so we can use it for hf.from_pretrained
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    # WEIGHTS_NAME = "pytorch_model.bin"
    WEIGHTS_NAME = "model.safetensors"
    # output_dir = os.path.join(args.output_dir, sub_folder)
    if not output_dir:
        os.makedirs(output_dir, exist_ok=True)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    for key in list(save_dict.keys()):
        if "lora" in key:
            del save_dict[key]
    # torch.save(save_dict, output_model_file)
    # 使用 safetensors 保存
    safe_save_file(save_dict, output_model_file, metadata={"format": "pt"})
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_pretrained(output_dir)

def format_seconds(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if secs > 0:
        parts.append(f"{secs} second{'s' if secs != 1 else ''}")
    return ', '.join(parts)

def plot_loss(losses, save_path):
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)

if __name__ == '__main__':
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(backend="nccl", init_method="env://")
    world_size = int(os.environ['WORLD_SIZE'])

    data_path = "./pt_sample_data.txt"
    data_split = "0.1"
    output_path = "./test_output"
    seed = 42
    max_seq_len = 128

    tokenizer = AutoTokenizer.from_pretrained('../llama_1B')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    train_data, eval_data = create_pretrain_dataset(
        local_rank, data_path, data_split, output_path, seed, tokenizer, max_seq_len
    )

    print(f"\n🔢 训练集大小: {len(train_data)}")
    if eval_data is not None:
        print(f"🔢 验证集大小: {len(eval_data)}")
    
    # 在数据加载后检查不同rank的数据差异
    if dist.get_rank() == 0:
        print("Rank0 数据示例:", train_data[0])
    dist.barrier()
    if dist.get_rank() == 1:
        print("Rank1 数据示例:", train_data[0])

    # # 一条样本
    # sample = train_data[10]
    # for k, v in sample.items():
    #     print(f"{k}: {v}")

    # # 解码后的文本
    # decoded = tokenizer.decode(sample['input_ids'])
    # print("\n📄 解码后的文本示例:")
    # print(decoded)
    # torchrun --nproc_per_node=2 utils.py

def plot_loss():
