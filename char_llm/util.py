import torch
import wandb


def sample(model, device: str, decode):
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
