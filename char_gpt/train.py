import argparse
from dataclasses import dataclass, asdict

import torch
import wandb

from char_gpt.model import GPTConfig, GPT


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


def load_data(name: str, device: str, block_size: int, batch_size: int):
    with open(name, 'r', encoding='utf-8') as f:
        text = f.read()

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    # data loading
    def get_batch(split):
        # generate a small batch of data of inputs x and targets y
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    return get_batch, encode, decode, vocab_size


@torch.no_grad()
def estimate_loss(model: GPT, get_batch, eval_iters: int):
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


def sample(model: GPT, device: str, decode):
    # generate from the model
    bs = 3
    max_new_tokens = 100
    context = torch.zeros((bs, 1), dtype=torch.long, device=device)
    return [decode(model.generate(context, max_new_tokens=max_new_tokens)[i].tolist()) for i in range(bs)]


def make_wandb_table(texts):
    columns = ["Text"]
    # Method 1
    data = [[text] for text in texts]
    table = wandb.Table(data=data, columns=columns)
    return table


def train(data_file: str, device: str, train_args: TrainArgs):
    get_batch, _, decode, vocab_size = load_data(
        data_file, device, train_args.block_size, train_args.batch_size)

    config = GPTConfig(
        block_size=train_args.block_size,
        vocab_size=vocab_size,
        n_layer=train_args.n_layer,
        n_head=train_args.n_head,
        n_embd=train_args.n_embd,
        dropout=train_args.dropout,
    )
    model = GPT(config).to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_args.learning_rate)

    for iter in range(train_args.max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % train_args.eval_interval == 0 or iter == train_args.max_iters - 1:
            losses = estimate_loss(
                model=model, get_batch=get_batch, eval_iters=train_args.eval_iters)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
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

    losses = estimate_loss(
        model=model, get_batch=get_batch, eval_iters=train_args.eval_iters)
    print(f"step {train_args.max_iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
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
    parser.add_argument('--eval_interval', type=int, required=True)
    parser.add_argument('--max_iters', type=int, default=5000)
    parser.add_argument('--wandb_project',
                        type=str, default='char-gpt')
    parser.add_argument('--data_file',
                        type=str, help='data filename')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    data_file = args.data_file
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_args = TrainArgs(
        eval_interval=args.eval_interval, max_iters=args.max_iters)
    wandb.init(project=args.wandb_project, config=asdict(train_args))
    train(data_file, device, train_args)


if __name__ == '__main__':
    main()
