import torch
from tqdm import tqdm

from ddpm.schedule import extract


# forward diffusion
def q_sample(
    x_start,
    t,
    noise,  # =None
    sqrt_alphas_cumprod,
    sqrt_one_minus_alphas_cumprod,
):
    # if noise is None:

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + \
        sqrt_one_minus_alphas_cumprod_t * noise


@torch.no_grad()
def p_sample(
    model,
    x,
    t,
    t_index,
    betas,
    sqrt_one_minus_alphas_cumprod,
    sqrt_recip_alphas,
    posterior_variance
):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

# Algorithm 2 but save all images:


@torch.no_grad()
def p_sample_loop(
    model, shape, timesteps,
    betas,
    sqrt_one_minus_alphas_cumprod,
    sqrt_recip_alphas,
    posterior_variance
) -> list[torch.Tensor]:
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(
            reversed(range(0, timesteps)),
            desc='sampling loop time step',
            total=timesteps):
        img = p_sample(
            model,
            img,
            torch.full((b,), i, device=device, dtype=torch.long),
            i,
            betas=betas,
            sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
            sqrt_recip_alphas=sqrt_recip_alphas,
            posterior_variance=posterior_variance,
        )
        imgs.append(img.cpu())
    return imgs


@torch.no_grad()
def sample(model,
           image_size,
           batch_size,  # =16,
           channels,  # =3,
           timesteps,
           betas,
           sqrt_one_minus_alphas_cumprod,
           sqrt_recip_alphas,
           posterior_variance
           ):
    return p_sample_loop(
        model, shape=(batch_size, channels, image_size, image_size),
        timesteps=timesteps,
        betas=betas,
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
        sqrt_recip_alphas=sqrt_recip_alphas,
        posterior_variance=posterior_variance,
    )
