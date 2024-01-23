import numpy as np
import torch
from torchvision.transforms import (CenterCrop, Compose, Lambda, Resize,
                                    ToPILImage, ToTensor)

from ddpm.sampler import q_sample


def make_transform(image_size):
    # image_size = 128
    transform = Compose([
        Resize(image_size),
        CenterCrop(image_size),
        ToTensor(),  # turn into Numpy array of shape HWC, divide by 255
        Lambda(lambda t: (t * 2) - 1),

    ])

    # x_start = transform(image).unsqueeze(0)
    # x_start.shape

    reverse_transform = Compose([
        Lambda(lambda t: (t + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])

    return transform, reverse_transform


def get_noisy_image(
    x_start,
    t,
    sqrt_alphas_cumprod,
    sqrt_one_minus_alphas_cumprod,
    reverse_transform,
):
    # add noise
    noise = torch.randn_like(x_start)
    x_noisy = q_sample(
        x_start, t=t, noise=noise,
        sqrt_alphas_cumprod=sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod)

    # turn back into PIL image
    noisy_image = reverse_transform(x_noisy.squeeze())

    return noisy_image
