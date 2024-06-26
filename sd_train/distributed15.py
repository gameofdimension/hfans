import functools
import os
import sys

import torch

try:
    import torch_npu  # type: ignore # noqa
except ImportError:
    pass
import torch.distributed as dist
from diffusers import DDPMScheduler, UNet2DConditionModel  # type: ignore
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
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
    distributed,
):
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(  # type: ignore # noqa
            dataset, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)  # type: ignore
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
        torch.npu.set_device(local_rank)  # type: ignore
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


def make_model(checkpoint: str, device, dp_type, dtype):
    noise_scheduler = DDPMScheduler.from_pretrained(
        checkpoint, subfolder="scheduler")
    unet: torch.nn.Module = UNet2DConditionModel.from_pretrained(  # type: ignore # noqa
        checkpoint, subfolder='unet').to(device)  # type: ignore
    unet.requires_grad_(True)
    unet.train()
    if dp_type == "ddp":
        unet = torch.nn.parallel.DistributedDataParallel(
            unet,
        )
    elif dp_type == "fsdp":
        assert dtype == torch.bfloat16
        my_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=20000
        )
        unet = FullyShardedDataParallel(
            unet,  # type: ignore
            auto_wrap_policy=my_auto_wrap_policy,
            mixed_precision=MixedPrecision(
                param_dtype=dtype, cast_forward_inputs=True),
        )
    return unet, noise_scheduler


def train(
        unet, optimizer, noise_scheduler,
        batch_size, device, dtype, dataloader):
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

        if dtype == torch.bfloat16:
            enabled = True
        elif dtype == torch.float32:
            enabled = False
        else:
            assert False, f"Unknown dtype: {dtype}"

        if device == 'cuda':
            with torch.cuda.amp.autocast(dtype=dtype, enabled=enabled):
                model_pred = unet(
                    noisy_model_input,
                    timesteps,
                    encoder_hidden_states,
                    return_dict=False,
                )[0]
        elif device == 'npu':
            with torch.npu.amp.autocast(dtype=dtype, enabled=enabled):  # type: ignore # noqa
                model_pred = unet(
                    noisy_model_input,
                    timesteps,
                    encoder_hidden_states,
                    return_dict=False,
                )[0]
        else:
            assert dtype == torch.float32
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
    if sys.argv[3] == 'bf16':
        dtype = torch.bfloat16
    elif sys.argv[3] == 'fp32':
        dtype = torch.float32
    else:
        assert False, f"Unknown dtype: {sys.argv[3]}"
    dp_type = None
    if len(sys.argv) > 4:
        dp_type = sys.argv[4]
        assert dp_type in ["ddp", "fsdp"]

    distributed = dp_type is not None
    if distributed:
        init_distributed(device)

    checkpoint = 'runwayml/stable-diffusion-v1-5'
    # checkpoint = '/root/model-repo/llm-stable-diffusion-v1-5'
    model, noise_scheduler = make_model(checkpoint, device, dp_type, dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    dataset = RandomDataset(size=1000000)
    dataloader = build_random_dataloader(batch_size, dataset, distributed)
    train(
        model, optimizer, noise_scheduler,
        batch_size, device, dtype, dataloader)

    if distributed:
        cleanup()


if __name__ == "__main__":
    """
    cuda:
        torchrun --nnodes=1 --nproc_per_node=4 --master_addr=localhost
        --master_port=30601 --node_rank=0 -m sd_train.distributed15 4 cuda
    npu:
        torchrun --nnodes=1 --nproc_per_node=8 --master_addr=localhost
        --master_port=30601 --node_rank=0 dist.py 4 npu
    """
    main()
