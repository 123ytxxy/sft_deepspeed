from torch.utils.data import DataLoader,RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from utils import create_pretrain_dataset, get_optimizer_grouped_parameters, get_train_ds_config, set_random_seed, plot_loss
from argparse import ArgumentParser
from transformers import AutoTokenizer
import torch
import torch.distributed as dist
import os
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from transformers import get_scheduler
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import math
import deepspeed
from utils import print_rank_0, to_device, save_hf_format, format_seconds
from sft_utils import build_instruction_dataset, DataCollatorForSupervisedDataset
from deepspeed import get_accelerator
import time

parser = ArgumentParser()
parser.add_argument('--local_rank', type=int, default=0, help='本地进程的排名')
parser.add_argument('--data_split', type=str, default='0.9,0.1', help='训练集、测试集的比例')
parser.add_argument('--data_output_path', type=str, default="./test_output", help='处理后的数据输出路径')
parser.add_argument('--data_path', type=str, default='./alpaca_data_zh_51k.json', help='数据输入路径')
parser.add_argument('--seed', type=int, default=42, help='随机种子值')
parser.add_argument('--max_seq_len', type=int, default=128, help='最大序列长度')
parser.add_argument('--per_device_train_batch_size', type=int, default=16, help='训练的batch')
parser.add_argument('--per_device_eval_batch_size', type=int, default=32, help='测试的batch')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='衰减率')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
parser.add_argument('--offload', type=bool, default=True, help='优化器')
parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
parser.add_argument('--lr_scheduler_type', type=str, default='linear')
parser.add_argument('--zero_stage', type=int, default=2)
parser.add_argument('--tensorboard_path', type=str, default='./')
parser.add_argument('--enable_tensorboard', type=bool, default=True)
parser.add_argument('--gradient_checkpointing', type=bool, default=True)
parser.add_argument('--output_dir', type=str, default='./model_py/')
parser.add_argument('--num_warmup_steps', type=int, default=1000)
parser.add_argument('--num_train_epochs', type=int, default=2)
parser.add_argument('--print_loss', type=bool, default=True)
parser.add_argument('--loss_save_path', type=str, default='./loss_png/loss_plot.png', help='损失图像保存地址')

def main():
    # 解析参数
    args = parser.parse_args()
    if args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        deepspeed.init_distributed()
    args.global_rank = torch.distributed.get_rank()
    ds_config = get_train_ds_config(offload=args.offload,
                                stage=args.zero_stage,
                                enable_tensorboard=args.enable_tensorboard,
                                tb_path=args.tensorboard_path,
                                tb_name="step1_model"
                                    )
    ds_config['train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    ds_config['train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size() * args.gradient_accumulation_steps
    
    torch.distributed.barrier()

    # 分词器导入
    tokenizer = AutoTokenizer.from_pretrained('../llama_1B')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'

    # 模型载入
    model_config = LlamaConfig.from_pretrained('../llama_1B')
    model = LlamaForCausalLM.from_pretrained('../llama_1B', config=model_config)
    # print(model.config)
    # 确保模型配置与 Tokenizer 一致
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    # 数据准备
    train_dataset = build_instruction_dataset(data_path=args.data_path, tokenizer=tokenizer,
                max_seq_length=args.max_seq_len, data_cache_dir = None,
                preprocessing_num_workers = 1,
                )
    # eval_dataset = None
    # 数据载入
    if args.local_rank ==-1:
        train_sampler =RandomSampler(train_dataset, shuffle=True, seed=args.seed)
        # eval_sampler =SequentialSampler(eval_dataset, shuffle=False)
    else:
        train_sampler =DistributedSampler(train_dataset, shuffle=True, seed=args.seed)
        # eval_sampler =DistributedSampler(eval_dataset, shuffle=False)
    
    collate_fn = DataCollatorForSupervisedDataset(tokenizer)
    train_dataloader=DataLoader(train_dataset,
                        collate_fn=collate_fn,
                        sampler=train_sampler,
                        batch_size=args.per_device_train_batch_size)
    # eval_dataloader =DataLoader(eval_dataset,
    #                     collate_fn=collate_fn,
    #                     sampler=eval_sampler,
    #                     batch_size=args.per_device_eval_batch_size)

    # 优化器配置
    # 参数分为两组，一组权重衰减，一组不衰减
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, args.weight_decay)
    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                                lr=args.learning_rate,
                                betas=(0.9, 0.95))
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    lr_scheduler = get_scheduler(
                    name=args.lr_scheduler_type,
                    optimizer=optimizer,
                    num_warmup_steps=args.num_warmup_steps,
                    num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
                    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=args,
            config=ds_config,
            lr_scheduler=lr_scheduler,
            dist_init_required=True)

    # 模型训练
    print_rank_0("***** Running training *****", args.global_rank)
    # perplexity = evaluation(model, eval_dataloader)
    # print_rank_0(f"ppl: {perplexity}", args.global_rank)

    losses = [] 
    train_start_time = time.time()
    for epoch in range(args.num_train_epochs):
        print_rank_0(
        f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, \
        Total Micro Batches {len(train_dataloader)}",
        args.global_rank)
        train_dataloader.sampler.set_epoch(epoch)  # shuffle的作用
        
        model.train()
        for step, batch in enumerate(train_dataloader):
            # 添加数据验证
            # print(f"Rank {dist.get_rank()} batch data sample: {batch}") 
            # 在训练开始验证打印某层权重
            # print(f"Rank {dist.get_rank()} weight before training:", model.lm_head.weight[0, :5])        
            start = time.time()
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            losses.append(loss.item())

            model.backward(loss)  # 执行反向传播
            # 验证梯度
            # print(f"Gradients:", model.module.layers.h[0].mlp.dense_h_to_4h.weight.grad.norm())
            model.step()          # 更新模型参数
            end = time.time()
            # 在训练结束时验证打印某层权重
            # print(f"Rank {dist.get_rank()} weight after training:", model.lm_head.weight[0, :5]) 
            if args.print_loss and step % 10 == 0:  # 每10步打印一次
                print(f"Epoch: {epoch}, Step: {step}, Rank: {dist.get_rank()}, loss = {loss.detach().float()}")
                print('Time:{}'.format(end - start))
    train_end_time = time.time()
    formatted_time = format_seconds(train_start_time - train_end_time)
    print_rank_0(f"Training completed successfully. Time spent: {formatted_time}.")
    plot_loss(losses, args.loss_save_path)
    if args.output_dir is not None:
        print_rank_0('saving the final model ...', args.global_rank)
        plot_loss()

    if args.global_rank == 0:
        save_hf_format(model, tokenizer, args.output_dir)
    
if __name__ == '__main__':
    main()