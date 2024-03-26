import functools
import os
import sys

import torch

from sd_train.common import SDXLModel

try:
    import torch_npu  # type: ignore # noqa
except ImportError:
    pass
import torch.distributed as dist
from diffusers import DDPMScheduler, UNet2DConditionModel  # type: ignore
from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (AutoTokenizer, CLIPTextModel,  # type: ignore # noqa
                          CLIPTextModelWithProjection)


class RandomDataset(Dataset):
    def __init__(self, size: int):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        model_input = torch.randn(
            (4, 64, 64))
        # prompt_embeds = torch.randn((77, 2048))
        return {
            "latents": model_input,
            # "prompt_embeds": prompt_embeds,
            "input_ids1": torch.randint(0, 10000, (77,)),
            "time_ids": torch.randn((6,)),
            # "text_embeds": torch.randn((1280,)),
            "input_ids2": torch.randint(0, 10000, (77,)),
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
    unet: torch.nn.Module = UNet2DConditionModel.from_pretrained(  # type: ignore # noqa
        checkpoint, subfolder='unet')  # .to(device)  # type: ignore
    # unet.requires_grad_(True)
    # unet.train()
    text_encoder1: torch.nn.Module = CLIPTextModel.from_pretrained(  # type: ignore # noqa
        checkpoint, subfolder="text_encoder"
    )  # .to(device)  # type: ignore
    # text_encoder1.requires_grad_(True)
    # text_encoder1.train()
    text_encoder2: torch.nn.Module = CLIPTextModelWithProjection.from_pretrained(  # type: ignore # noqa
        checkpoint, subfolder="text_encoder_2"
    )  # .to(device)  # type: ignore
    # text_encoder2.requires_grad_(True)
    # text_encoder2.train()
    noise_scheduler = DDPMScheduler.from_pretrained(
        checkpoint, subfolder="scheduler")

    tokenizer1 = AutoTokenizer.from_pretrained(
        checkpoint,
        subfolder="tokenizer",
    )
    tokenizer2 = AutoTokenizer.from_pretrained(
        checkpoint,
        subfolder="tokenizer_2",
    )

    sdxl_model = SDXLModel(
        unet,  text_encoder1, text_encoder2).to(device)
    sdxl_model = sdxl_model.requires_grad_(True)
    sdxl_model.train()

    if dp_type == "ddp":
        sdxl_model = torch.nn.parallel.DistributedDataParallel(
            sdxl_model,
            find_unused_parameters=True,
        )
    elif dp_type == "fsdp":
        assert dtype == torch.bfloat16
        my_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=20000
        )
        sdxl_model = FullyShardedDataParallel(
            sdxl_model,  # type: ignore
            auto_wrap_policy=my_auto_wrap_policy,
            mixed_precision=MixedPrecision(
                param_dtype=dtype, cast_forward_inputs=True),
        )

    return sdxl_model, noise_scheduler, tokenizer1, tokenizer2


def train(
        model: SDXLModel, optimizer, noise_scheduler,
        batch_size, tokenizer1, tokenizer2, device, dtype, dataloader):

    model_max_length1 = tokenizer1.model_max_length
    model_max_length2 = tokenizer2.model_max_length
    eos_token_id2 = tokenizer2.eos_token_id
    for batch in tqdm(dataloader):
        latents = batch['latents'].to(device)
        input_ids1 = batch['input_ids1'].to(device)
        input_ids2 = batch['input_ids2'].to(device)
        time_ids = batch['time_ids'].to(device)

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
                model_pred = model(
                    timesteps, input_ids1, input_ids2,
                    noisy_model_input, time_ids,
                    model_max_length1, model_max_length2,
                    eos_token_id2)
        elif device == 'npu':
            with torch.npu.amp.autocast(dtype=dtype, enabled=enabled):  # type: ignore # noqa
                model_pred = model(
                    timesteps, input_ids1, input_ids2,
                    noisy_model_input, time_ids,
                    model_max_length1, model_max_length2,
                    eos_token_id2)
        else:
            assert dtype == torch.float32
            model_pred = model(
                timesteps, input_ids1, input_ids2,
                noisy_model_input, time_ids,
                model_max_length1, model_max_length2,
                eos_token_id2)

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

    checkpoint = 'stabilityai/stable-diffusion-xl-base-1.0'
    # checkpoint = '/root/model-repo/llm-stable-diffusion-xl-base-1.0'
    model, noise_scheduler, tokenizer1, tokenizer2 = make_model(
        checkpoint, device, dp_type, dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, foreach=False)
    dataset = RandomDataset(size=1000000)
    dataloader = build_random_dataloader(batch_size, dataset, distributed)
    train(
        model, optimizer, noise_scheduler,  # type: ignore
        batch_size, tokenizer1, tokenizer2,
        device, dtype, dataloader)

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
