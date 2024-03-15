import functools
import os
import sys

import torch
import torch.distributed as dist
from diffusers import DDPMScheduler, UNet2DConditionModel  # type: ignore
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from tqdm import tqdm


def cleanup():
    dist.destroy_process_group()


def init_distributed():

    # Initializes the distributed backend
    # which will take care of sychronizing nodes/GPUs
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
    return world_size, rank, local_rank


def make_model(checkpoint: str, device, dtype):
    unet: torch.nn.Module = UNet2DConditionModel.from_pretrained(  # type: ignore # noqa
        checkpoint, subfolder='unet').to(device)  # type: ignore
    unet.requires_grad_(True)
    unet.train()
    noise_scheduler = DDPMScheduler.from_pretrained(
        checkpoint, subfolder="scheduler")
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=20000
    )
    unet = FSDP(
        unet,  # type: ignore
        auto_wrap_policy=my_auto_wrap_policy,
        mixed_precision=MixedPrecision(
            param_dtype=dtype, cast_forward_inputs=True),
    )
    return unet, noise_scheduler


def batch_data(noise_scheduler, batch_size, device):
    model_input = torch.randn(
        (batch_size, 4, 64, 64), device=device)
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps,
        (batch_size,), device=device
    )
    noise = torch.randn_like(model_input)
    noisy_model_input = noise_scheduler.add_noise(
        model_input, noise, timesteps)
    prompt_embeds = torch.rand(
        (batch_size, 77, 2048), device=device)
    unet_added_conditions = {
        "time_ids": torch.rand((batch_size, 6), device=device),
        "text_embeds": torch.rand((batch_size, 1280), device=device),
    }
    return (
        noisy_model_input, timesteps, prompt_embeds,
        unet_added_conditions, noise
    )


def train(unet, optimizer, noise_scheduler, batch_size, device, dtype):
    total_step = 1000000
    for step in tqdm(range(total_step)):

        batch = batch_data(noise_scheduler, batch_size, device)
        noisy_model_input, timesteps, prompt_embeds, \
            unet_added_conditions, noise = batch

        with torch.autocast(
                device_type=device, dtype=dtype, enabled=True):
            model_pred = unet(
                noisy_model_input,
                timesteps,
                prompt_embeds,
                added_cond_kwargs=unet_added_conditions,
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
    device = 'cuda'
    dtype = torch.bfloat16
    init_distributed()

    model, noise_scheduler = make_model(
        "stabilityai/stable-diffusion-xl-base-1.0", device, dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train(model, optimizer, noise_scheduler, batch_size, device, dtype)

    cleanup()


if __name__ == "__main__":
    main()
