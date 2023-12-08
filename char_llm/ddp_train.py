import argparse
from dataclasses import asdict
import os
from typing import Any
from loguru import logger
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

from char_llm.model_gpt import GPT, GPTConfig
from char_llm.util import TrainArgs


def cleanup():
    dist.destroy_process_group()


def init_distributed():

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist_url = "env://"  # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    dist.init_process_group(backend="nccl",
                            init_method=dist_url,
                            world_size=world_size,
                            rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)
    # synchronizes all the threads to reach this point before moving on
    dist.barrier()

    def setup_for_distributed(is_master):
        """
        This function disables printing when not in master process
        """
        import builtins as __builtin__
        builtin_print = __builtin__.print

        def print(*args, **kwargs):
            force = kwargs.pop('force', False)
            if is_master or force:
                builtin_print(*args, **kwargs)

        __builtin__.print = print

    setup_for_distributed(rank == 0)


class CharSeqDataset(Dataset):

    def __init__(self, token_ids: torch.Tensor, seq_len: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.token_ids = token_ids

    def __len__(self):
        return self.token_ids.size(0) - self.seq_len

    def __getitem__(self, index) -> Any:
        assert 0 <= index < self.__len__()
        return self.token_ids[index:index + self.seq_len + 1]


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset,
                                                               shuffle=shuffle)
    if shuffle:
        return torch.utils.data.RandomSampler(dataset)
    else:
        return torch.utils.data.SequentialSampler(dataset)


def create_dataloader(instance: Dataset, is_inference: bool, distributed: bool,
                      batch_size: int):
    dataloader = torch.utils.data.DataLoader(
        instance,
        batch_size=batch_size,
        sampler=data_sampler(instance,
                             shuffle=not is_inference,
                             distributed=distributed),
        drop_last=not is_inference,
        num_workers=1,
    )

    return dataloader


def make_dataset(name: str, block_size: int):
    with open(name, 'r', encoding='utf-8') as f:
        text = f.read()

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s
                        ]  # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[
        i] for i in l])  # decoder: take a list of integers, output a string

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    return vocab_size, decode, CharSeqDataset(train_data,
                                              block_size), CharSeqDataset(
                                                  val_data, block_size)


def prepare_dataloader(name: str, block_size: int, batch_size: int):
    vocab_size, decode, train_ds, val_ds = make_dataset(name, block_size)
    train_dl = create_dataloader(train_ds, False, True, batch_size)
    test_dl = create_dataloader(val_ds, True, True, batch_size)
    return vocab_size, decode, train_dl, test_dl


def build_model(train_args: TrainArgs, vocab_size: int, device):
    config = GPTConfig(
        block_size=train_args.block_size,
        vocab_size=vocab_size,
        n_layer=train_args.n_layer,
        n_head=train_args.n_head,
        n_embd=train_args.n_embd,
        dropout=train_args.dropout,
    )
    logger.info(f"gpt config {asdict(config)}")
    return GPT(config).to(device)


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
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for i in range(max_epoch):
        for batch in train_dl:
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
        n_embd=4096,
        n_layer=32,
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
