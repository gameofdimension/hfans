import argparse
from dataclasses import dataclass, asdict

import torch
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR

from char_llm.data import load_data
from char_llm.model_gpt import GPTConfig, GPT
from char_llm.model_llama import LlamaConfig, CausalLlamaModel
from char_llm.model_rwkv import RwkvConfig, CausalRwkvModel
from char_llm.util import sample, make_wandb_table

from loguru import logger


@dataclass
class TrainArgs:
    # hyperparameters
    batch_size: int = 64  # how many independent sequences will we process in parallel?
    block_size: int = 256  # what is the maximum context length for predictions?
    max_iters: int = 5000
    eval_interval: int = 500
    learning_rate: float = 3e-4
    eval_iters: int = 200
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: int = 0.2


@torch.no_grad()
def estimate_loss(model, get_batch, eval_iters: int):
    """

    :param model: gpt or rwkv model
    :param get_batch:
    :param eval_iters:
    :return:
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def build_model(train_args: TrainArgs, model_type: str, device, vocab_size):
    if model_type == 'gpt':
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
    if model_type == 'rwkv':
        train_args.n_embd = 128
        train_args.n_layer = 2
        train_args.learning_rate = 1e-3
        train_args.block_size = 64
        config = RwkvConfig(
            hidden_size=train_args.n_embd,
            attention_hidden_size=train_args.n_embd,
            intermediate_size=train_args.n_embd * 2,
            context_length=train_args.block_size,
            num_hidden_layers=train_args.n_layer,
            vocab_size=vocab_size,
        )
        logger.info(f"rwkv config {asdict(config)}")
        return CausalRwkvModel(config).to(device)
    if model_type == 'llama':
        config = LlamaConfig(
            max_context_length=train_args.block_size,
            vocab_size=vocab_size,
            num_hidden_layers=train_args.n_layer,
            num_attention_heads=train_args.n_head,
            hidden_size=train_args.n_embd,
            intermediate_size=int(2.6875 * train_args.n_embd)
        )
        logger.info(f"llama config {asdict(config)}")
        return CausalLlamaModel(config).to(device)
    assert False, f"unknown model type {model_type}"


def train(data_file: str, device: str, model_type: str, train_args: TrainArgs):
    get_batch, _, decode, vocab_size = load_data(
        data_file, device, train_args.block_size, train_args.batch_size)

    model = build_model(train_args, model_type, device, vocab_size)
    # print the number of parameters in the model
    logger.info(f"training args {train_args}")
    logger.info(f'{sum(p.numel() for p in model.parameters()) / 1e6} M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_args.learning_rate)
    # scheduler = CosineAnnealingLR(optimizer, T_max=train_args.max_iters)

    for iter in range(train_args.max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % train_args.eval_interval == 0 or iter == train_args.max_iters - 1:
            losses = estimate_loss(
                model=model, get_batch=get_batch, eval_iters=train_args.eval_iters)
            logger.info(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            texts = sample(model, device, decode)
            wandb.log(
                step=iter,
                commit=True,
                data={
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "samples": make_wandb_table(texts),
                })

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        # scheduler.step()

    losses = estimate_loss(
        model=model, get_batch=get_batch, eval_iters=train_args.eval_iters)
    logger.info(f"step {train_args.max_iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    texts = sample(model, device, decode)
    wandb.log(
        step=train_args.max_iters,
        commit=True,
        data={
            "train/loss": losses['train'],
            "val/loss": losses['val'],
            "samples": make_wandb_table(texts),
        })


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_interval', type=int, default=50)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--max_iters', type=int, default=5000)
    parser.add_argument('--model_type', type=str, required=True, help='gpt or rwkv or llama')
    parser.add_argument('--wandb_project', type=str, required=True)
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--data_file', type=str, help='data filename')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    data_file = args.data_file
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_args = TrainArgs(
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
        max_iters=args.max_iters,
        learning_rate=args.lr,
    )
    wandb.init(project=args.wandb_project, config=asdict(train_args))
    train(data_file, device, args.model_type, train_args)


if __name__ == '__main__':
    main()
