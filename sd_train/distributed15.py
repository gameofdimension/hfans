import os
import sys

import torch
try:
    import torch_npu  # type: ignore # noqa
except ImportError:
    pass
import torch.distributed as dist
from diffusers import DDPMScheduler, UNet2DConditionModel  # type: ignore
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class RandomDataset(Dataset):
    def __init__(self, size: int):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        latents = torch.randn(
            (4, 64, 64))
        encoder_hidden_states = torch.randn((77, 768))
        return {
            "latents": latents,
            "encoder_hidden_states": encoder_hidden_states,
        }


def build_random_dataloader(
    batch_size: int,
    dataset,
):
    sampler = torch.utils.data.distributed.DistributedSampler(  # type: ignore
        dataset, shuffle=False)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=True,
        sampler=sampler,
        num_workers=4,
    )

    return dataloader


def cleanup():
    dist.destroy_process_group()


def init_distributed(device):

    # Initializes the distributed backend
    # which will take care of sychronizing nodes/GPUs
    dist_url = "env://"  # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    if device == 'cuda':
        dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)
        # this will make all .cuda() calls work properly
        torch.cuda.set_device(local_rank)
    elif device == 'npu':
        dist.init_process_group(
            backend="hccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)
        torch.npu.set_device(local_rank)
    elif device == 'cpu':
        dist.init_process_group(
            backend="gloo",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)
        torch.cpu.set_device(local_rank)
    else:
        assert False, f"Unknown device: {device}"
    # synchronizes all the threads to reach this point before moving on
    dist.barrier()
    return world_size, rank, local_rank


def make_model(checkpoint: str, device, local_rank):
    unet: torch.nn.Module = UNet2DConditionModel.from_pretrained(  # type: ignore # noqa
        checkpoint, subfolder='unet').to(device)  # type: ignore
    unet.requires_grad_(True)
    unet.train()
    unet = torch.nn.parallel.DistributedDataParallel(
        unet,
        # device_ids=[local_rank],
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        checkpoint, subfolder="scheduler")
    return unet, noise_scheduler


def train(unet, optimizer, noise_scheduler, batch_size, device):
    dataset = RandomDataset(
        size=1000000,
    )
    dataloader = build_random_dataloader(batch_size, dataset)
    for batch in tqdm(dataloader):
        latents = batch['latents'].to(device)
        encoder_hidden_states = batch['encoder_hidden_states'].to(device)

        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=device
        )
        noise = torch.randn_like(latents)
        noisy_model_input = noise_scheduler.add_noise(
            latents, noise, timesteps)

        model_pred = unet(
            noisy_model_input,
            timesteps,
            encoder_hidden_states,
            return_dict=False,
        )[0]

        assert noise_scheduler.config.prediction_type == "epsilon"
        loss = torch.nn.functional.mse_loss(
            model_pred.float(), noise.float(), reduction="mean")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def main():
    lr = 2e-5
    batch_size = int(sys.argv[1])
    device = sys.argv[2]
    world_size, rank, local_rank = init_distributed(device)

    checkpoint = 'runwayml/stable-diffusion-v1-5'
    # checkpoint = '/root/model-repo/llm-stable-diffusion-v1-5'
    model, noise_scheduler = make_model(checkpoint, device, local_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train(model, optimizer, noise_scheduler, batch_size, device)

    cleanup()


if __name__ == "__main__":
    """
    cuda:
        torchrun --nnodes=1 --nproc_per_node=4 --master_addr=localhost --master_port=30601 --node_rank=0 -m sd_train.distributed15 4 cuda
    npu:
        torchrun --nnodes=1 --nproc_per_node=8 --master_addr=localhost --master_port=30601 --node_rank=0 dist.py 4 npu
    """
    main()
