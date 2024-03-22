import sys

import torch
from diffusers import DDPMScheduler, UNet2DConditionModel  # type: ignore
from tqdm import tqdm


def make_model(checkpoint: str, device):
    unet: torch.nn.Module = UNet2DConditionModel.from_pretrained(  # type: ignore # noqa
        checkpoint, subfolder='unet').to(device)  # type: ignore
    unet.requires_grad_(True)
    unet.train()
    noise_scheduler = DDPMScheduler.from_pretrained(
        checkpoint, subfolder="scheduler")
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


def train(unet, optimizer, noise_scheduler, batch_size, device):
    total_step = 1000000
    for step in tqdm(range(total_step)):

        batch = batch_data(noise_scheduler, batch_size, device)
        noisy_model_input, timesteps, prompt_embeds, \
            unet_added_conditions, noise = batch

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
    device = sys.argv[2]
    checkpoint = 'stabilityai/stable-diffusion-xl-base-1.0'
    # checkpoint = '/root/model-repo/llm-stable-diffusion-xl-base-1.0'
    model, noise_scheduler = make_model(checkpoint, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train(model, optimizer, noise_scheduler, batch_size, device)


if __name__ == "__main__":
    main()
