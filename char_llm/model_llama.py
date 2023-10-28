from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.activations import get_activation

from char_llm.common import RMSNorm, precompute_cos_sin, apply_rotary


@dataclass
class LlamaConfig:
    hidden_size: int = 4096
    vocab_size: int = 32000
    rms_norm_eps: float = 1e-05
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    intermediate_size: int = 11008
    max_context_length: int = 4096
    hidden_act: str = "silu"


class Mlp(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(in_features=config.hidden_size, out_features=config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(in_features=config.hidden_size, out_features=config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(in_features=config.intermediate_size, out_features=config.hidden_size, bias=False)
        self.act_fn = get_activation(config.hidden_act)

    def forward(self, hidden_states: torch.Tensor):
        assert hidden_states.dim() == 3
        activation = self.act_fn(self.gate_proj(hidden_states))
        hidden_states = activation * self.up_proj(hidden_states)
        hidden_states = self.down_proj(hidden_states)
        return hidden_states


class MultiHeadAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.q_proj = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
        self.k_proj = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
        self.v_proj = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
        self.o_proj = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)

        self.cos_sin_factory = precompute_cos_sin(
            config.max_context_length,
            config.hidden_size // config.num_attention_heads)
        self.config = config

    def forward(self, hidden_states: torch.Tensor):
        # [batch size, seq_length, hidden_size]
        assert hidden_states.dim() == 3
        bs, sl, hs = hidden_states.size()
        all_q = self.q_proj(hidden_states)
        all_k = self.k_proj(hidden_states)
        all_v = self.v_proj(hidden_states)

        head_num = self.config.num_attention_heads
        head_dim = self.config.hidden_size // head_num

        all_q = all_q.reshape(bs, sl, head_num, head_dim).permute(1, 0, 2, 3)
        all_k = all_k.reshape(bs, sl, head_num, head_dim).permute(1, 0, 2, 3)
        all_v = all_v.reshape(bs, sl, head_num, head_dim).permute(1, 0, 2, 3)

        for s in range(sl):
            cos, sin = self.cos_sin_factory(s)
            all_q[s] = apply_rotary(all_q[s], cos, sin)
            all_k[s] = apply_rotary(all_k[s], cos, sin)

        all_q = all_q.permute(1, 2, 0, 3)
        all_k = all_k.permute(1, 2, 0, 3)
        all_v = all_v.permute(1, 2, 0, 3)

        output = torch.nn.functional.scaled_dot_product_attention(all_q, all_k, all_v, is_causal=True)
        output = output.permute(0, 2, 1, 3).reshape(bs, sl, hs)
        return self.o_proj(output)


class Block(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.rn1 = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.mha = MultiHeadAttention(config)

        self.rn2 = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Mlp(config)

    def forward(self, hidden_states: torch.Tensor):
        """

        :param hidden_states: [seq_length, hidden_size]
        :return:
        """
        x = hidden_states
        for f, g in zip([self.mha, self.mlp], [self.rn1, self.rn2]):
            gx = g(x)
            fx = f(gx)
            x = x + fx
        return x


def name_mapping(param: str):
    out = {
        "word_embedding_table.weight": "model.embed_tokens.weight",
        "rms.weight": "model.norm.weight",
    }
    if param in out:
        return out[param]

    li = param.split('.')[1]
    prefix = f"model.layers.{li}."
    if "rn1.weight" in param:
        postfix = "input_layernorm.weight"
    elif "mha.q_proj.weight" in param:
        postfix = "self_attn.q_proj.weight"
    elif "mha.k_proj.weight" in param:
        postfix = "self_attn.k_proj.weight"
    elif "mha.v_proj.weight" in param:
        postfix = "self_attn.v_proj.weight"
    elif "mha.o_proj.weight" in param:
        postfix = "self_attn.o_proj.weight"
    elif "rn2.weight" in param:
        postfix = "post_attention_layernorm.weight"
    elif "mlp.gate_proj.weight" in param:
        postfix = "mlp.gate_proj.weight"
    elif "mlp.up_proj.weight" in param:
        postfix = "mlp.up_proj.weight"
    elif "mlp.down_proj.weight" in param:
        postfix = "mlp.down_proj.weight"
    else:
        assert False

    return prefix + postfix


class Model(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.word_embedding_table = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])
        self.rms = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.LongTensor):
        # [batch size, seq length]
        assert input_ids.dim() == 2
        hidden_states = self.word_embedding_table(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.rms(hidden_states)

    def load_weights_from_hf(self, model_id):
        """
        :return:
        """
        # model_id = 'felixdae/Llama-2-7b-hf'
        ref_model = AutoModelForCausalLM.from_pretrained(model_id)

        state_dict = self.state_dict()
        ref_state_dict = ref_model.state_dict()
        for tup in self.named_parameters():
            name = tup[0]
            param = state_dict[name]
            ref_name = name_mapping(name)
            ref_param = ref_state_dict[ref_name]
            param.data.copy_(ref_param)


if __name__ == '__main__':
    config = LlamaConfig(num_hidden_layers=2)
    mm = Model(config)

    mm(torch.LongTensor([[2, 3, 4], [34, 56, 78]]))
