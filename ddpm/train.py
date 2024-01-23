import itertools
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import (Compose, Lambda, RandomHorizontalFlip,
                                    ToTensor)
from torchvision.utils import save_image

from ddpm.model import Unet
from ddpm.sampler import q_sample, sample
from ddpm.schedule import beta_dependents


def make_dataloader(image_size, channels, batch_size):
    # load dataset from the hub
    dataset = load_dataset("fashion_mnist")

    # define image transformations (e.g. using torchvision)
    transform = Compose([
        RandomHorizontalFlip(),
        ToTensor(),
        Lambda(lambda t: (t * 2) - 1)
    ])

    # define function

    def transforms(examples):
        examples["pixel_values"] = [
            transform(image.convert("L")) for image in examples["image"]]
        del examples["image"]

        return examples

    transformed_dataset = dataset.with_transform(  # type: ignore
        transforms).remove_columns("label")

    # create dataloader
    dataloader = DataLoader(
        transformed_dataset["train"],  # type: ignore
        batch_size=batch_size, shuffle=True)
    return dataloader


def p_losses(
    denoise_model,
    x_start,
    t,
    noise,  # =None,
    loss_type,  # ="l1"
    sqrt_alphas_cumprod,
    sqrt_one_minus_alphas_cumprod,
):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(
        x_start=x_start, t=t,
        noise=noise, sqrt_alphas_cumprod=sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
    )
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def train():
    image_size = 28
    channels = 1
    batch_size = 128

    timesteps = 200
    (
        betas,
        alphas,
        alphas_cumprod,
        alphas_cumprod_prev,
        sqrt_recip_alphas,
        sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod,
        posterior_variance
    ) = beta_dependents(timesteps)

    results_folder = Path("./results")
    results_folder.mkdir(exist_ok=True)
    save_and_sample_every = 1000

    dataloader = make_dataloader(image_size, channels, batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1, 2, 4,)
    )
    model.to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)

    epochs = 5
    step = 0
    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            batch_size = batch["pixel_values"].shape[0]
            batch = batch["pixel_values"].to(device)

            # Algorithm 1 line 3: sample t uniformally
            # for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,),
                              device=device).long()

            loss = p_losses(
                model, batch, t,
                loss_type="huber",
                noise=None,
                sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod
            )

            if step % 100 == 0:
                print("Loss:", loss.item())

            loss.backward()
            optimizer.step()

            # save generated images
            if step != 0 and step % save_and_sample_every == 0:
                milestone = step // save_and_sample_every
                batches = num_to_groups(4, batch_size)
                all_images_list = list(map(lambda n: sample(
                    model,
                    batch_size=n,
                    channels=channels,
                    image_size=image_size,
                    timesteps=timesteps,
                    betas=betas,
                    sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,  # noqa
                    sqrt_recip_alphas=sqrt_recip_alphas,
                    posterior_variance=posterior_variance,
                ), batches))

                nrow = 5
                stride = timesteps//nrow
                flatten = itertools.chain.from_iterable
                all_images_list = list(
                    flatten([lst[::stride] for lst in all_images_list]))
                all_images = torch.cat(all_images_list, dim=0)
                all_images = (all_images + 1) * 0.5
                save_image(
                    all_images,
                    str(results_folder / f'sample-{milestone}.png'),
                    nrow=4)

            step += 1


if __name__ == "__main__":
    train()
