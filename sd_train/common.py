import torch
from transformers import (
    CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
)


class SDXLModel(torch.nn.Module):
    def __init__(
            self, unet: torch.nn.Module,
            text_encoder_one: torch.nn.Module,
            text_encoder_two: torch.nn.Module):
        super(SDXLModel, self).__init__()
        self.unet = unet
        self.text_encoder_one = text_encoder_one
        self.text_encoder_two = text_encoder_two

    def forward(
            self, timesteps: torch.Tensor, input_ids1: torch.Tensor,
            input_ids2: torch.Tensor, noisy_latents: torch.Tensor):
        pass


def pool_workaround(
    text_encoder: CLIPTextModelWithProjection,
    last_hidden_state: torch.Tensor,
    input_ids: torch.Tensor,
    eos_token_id: int
):
    # Create a mask where the EOS tokens are
    eos_token_mask = (input_ids == eos_token_id).int()

    # this will be 0 if there is no EOS token, it's fine
    eos_token_index = torch.argmax(eos_token_mask, dim=1)
    eos_token_index = eos_token_index.to(device=last_hidden_state.device)

    # get hidden states for EOS token
    pooled_output = last_hidden_state[torch.arange(
        last_hidden_state.shape[0],
        device=last_hidden_state.device), eos_token_index]

    pooled_output = text_encoder.text_projection(
        pooled_output.to(text_encoder.text_projection.weight.dtype))
    pooled_output = pooled_output.to(last_hidden_state.dtype)

    return pooled_output


def get_hidden_states_sdxl(
    input_ids1: torch.Tensor,
    input_ids2: torch.Tensor,
    tokenizer1: CLIPTokenizer,
    tokenizer2: CLIPTokenizer,
    text_encoder1: CLIPTextModel,
    text_encoder2: CLIPTextModelWithProjection,
    dtype,
):
    # input_ids: b,n,77 -> b*n, 77
    b_size = input_ids1.size()[0]
    input_ids1 = input_ids1.reshape(
        (-1, tokenizer1.model_max_length))  # batch_size*n, 77
    input_ids2 = input_ids2.reshape(
        (-1, tokenizer2.model_max_length))  # batch_size*n, 77

    # text_encoder1
    enc_out = text_encoder1(
        input_ids1, output_hidden_states=True, return_dict=True)
    hidden_states1 = enc_out["hidden_states"][11]

    # text_encoder2
    enc_out = text_encoder2(
        input_ids2, output_hidden_states=True, return_dict=True)
    hidden_states2 = enc_out["hidden_states"][-2]  # penuultimate layer

    # pool2 = enc_out["text_embeds"]
    pool2 = pool_workaround(
        text_encoder2, enc_out["last_hidden_state"],
        input_ids2, tokenizer2.eos_token_id)  # type: ignore

    # b*n, 77, 768 or 1280 -> b, n*77, 768 or 1280
    # n_size = 1 if max_token_length is None else max_token_length // (
    #     tokenizer1.model_max_length - 2)
    hidden_states1 = hidden_states1.reshape(
        (b_size, -1, hidden_states1.shape[-1]))
    hidden_states2 = hidden_states2.reshape(
        (b_size, -1, hidden_states2.shape[-1]))

    hidden_states1 = hidden_states1.to(dtype)
    hidden_states2 = hidden_states2.to(dtype)

    return hidden_states1, hidden_states2, pool2
