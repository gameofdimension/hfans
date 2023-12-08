import torch
import wandb
from dataclasses import dataclass

def sample(model, device: str, decode):
    # generate from the model
    bs = 3
    max_new_tokens = 100
    context = torch.zeros((bs, 1), dtype=torch.long, device=device)
    model.eval()
    res = [decode(model.generate(context, max_new_tokens=max_new_tokens)[i].tolist()) for i in range(bs)]
    model.train()
    return res


def make_wandb_table(texts):
    columns = ["Text"]
    # Method 1
    data = [[text] for text in texts]
    table = wandb.Table(data=data, columns=columns)
    return table

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
