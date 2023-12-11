import argparse
import functools
import os

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data import DataLoader
from tqdm import tqdm

from char_llm.distributed import (
    build_model,
    cleanup,
    init_distributed,
    prepare_dataloader,
)
from char_llm.model_gpt import GPT
from char_llm.util import TrainArgs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--max_epoch", type=int, default=500)
    parser.add_argument("--wandb_project", type=str, required=True)
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--data_file", type=str, help="data filename")
    args = parser.parse_args()
    return args


def train(max_epoch: int, train_dl: DataLoader, model: GPT, lr: float, device):
    # local_rank = int(os.environ["LOCAL_RANK"])
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=20000
    )
    model = FSDP(
        model,
        auto_wrap_policy=my_auto_wrap_policy,
        #  cpu_offload=CPUOffload(offload_params=True),
    )

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
    device = "cuda"
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
        data_file, train_args.block_size, train_args.batch_size
    )
    model = build_model(train_args, vocab_size, device)
    train(args.max_epoch, train_dl, model, train_args.learning_rate, device)

    cleanup()


if __name__ == "__main__":
    main()
