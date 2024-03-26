import sys

import torch
from diffusers import (DDPMScheduler, UNet2DConditionModel)  # type: ignore
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, AutoTokenizer  # type: ignore # noqa

from sd_train.common import SDXLModel


def make_model(checkpoint: str, device):
    unet: torch.nn.Module = UNet2DConditionModel.from_pretrained(  # type: ignore # noqa
        checkpoint, subfolder='unet').to(device)  # type: ignore
    unet.requires_grad_(True)
    unet.train()
    text_encoder1: torch.nn.Module = CLIPTextModel.from_pretrained(  # type: ignore # noqa
        checkpoint, subfolder="text_encoder"
    ).to(device)  # type: ignore
    text_encoder1.requires_grad_(True)
    text_encoder1.train()
    text_encoder2: torch.nn.Module = CLIPTextModelWithProjection.from_pretrained(  # type: ignore # noqa
        checkpoint, subfolder="text_encoder_2"
    ).to(device)  # type: ignore
    text_encoder2.requires_grad_(True)
    text_encoder2.train()
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
        unet, tokenizer1, tokenizer2,
        text_encoder1, text_encoder2)
    return sdxl_model, noise_scheduler


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
    input_ids1 = torch.randint(0, 10000, (batch_size, 77), device=device)
    input_ids2 = torch.randint(0, 10000, (batch_size, 77), device=device)
    time_ids = torch.randn((batch_size, 6), device=device)
    return (
        noisy_model_input,
        timesteps,
        input_ids1,
        input_ids2,
        time_ids,
        noise
    )


def train(model: SDXLModel, optimizer, noise_scheduler, batch_size, device):
    total_step = 1000000
    for step in tqdm(range(total_step)):

        batch = batch_data(noise_scheduler, batch_size, device)
        noisy_model_input, timesteps, input_ids1, \
            input_ids2, time_ids, noise = batch

        model_pred = model(
            timesteps, input_ids1, input_ids2, noisy_model_input, time_ids)

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
    checkpoint = 'stabilityai/stable-diffusion-xl-base-1.0'
    # checkpoint = '/root/model-repo/llm-stable-diffusion-xl-base-1.0'
    model, noise_scheduler = make_model(checkpoint, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train(model, optimizer, noise_scheduler, batch_size, device)


if __name__ == "__main__":
    main()
