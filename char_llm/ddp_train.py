import argparse
from dataclasses import asdict
import os
from typing import Any
from loguru import logger
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm
from char_llm.distributed import build_model, cleanup, init_distributed, prepare_dataloader

from char_llm.model_gpt import GPT, GPTConfig
from char_llm.util import TrainArgs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_interval', type=int, default=50)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--wandb_project', type=str, required=True)
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--data_file', type=str, help='data filename')
    args = parser.parse_args()
    return args


def train(max_epoch: int, train_dl: DataLoader, model: GPT, lr: float, device):
    local_rank = int(os.environ['LOCAL_RANK'])
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[local_rank],
                                                find_unused_parameters=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for i in range(max_epoch):
        for batch in tqdm(train_dl):
            batch = batch.to(device)
            X = batch[:, :-1].contiguous()
            Y = batch[:, 1:].contiguous()
            logits, loss = model(X, Y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


def main():
    args = get_args()
    data_file = args.data_file
    device = 'cuda'
    train_args = TrainArgs(
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
        learning_rate=args.lr,
        n_embd=8192,
        n_layer=4,
        n_head=32,
    )

    init_distributed()

    vocab_size, decode, train_dl, test_dl = prepare_dataloader(
        data_file, train_args.block_size, train_args.batch_size)
    model = build_model(train_args, vocab_size, device)
    train(args.max_epoch, train_dl, model, train_args.learning_rate, device)

    cleanup()


if __name__ == '__main__':
    main()
