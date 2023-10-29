from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
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
    initializer_range: float = 0.02
    dropout: float = 0.2


class Mlp(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(in_features=config.hidden_size, out_features=config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(in_features=config.hidden_size, out_features=config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(in_features=config.intermediate_size, out_features=config.hidden_size, bias=False)
        self.act_fn = get_activation(config.hidden_act)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor):
        assert hidden_states.dim() == 3
        activation = self.act_fn(self.gate_proj(hidden_states))
        hidden_states = activation * self.up_proj(hidden_states)
        hidden_states = self.down_proj(hidden_states)
        return self.dropout(hidden_states)


class MultiHeadAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.q_proj = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
        self.k_proj = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
        self.v_proj = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
        self.o_proj = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

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

        cos, sin = self.cos_sin_factory(hidden_states.device)
        all_q = apply_rotary(all_q, cos[:sl], sin[:sl])
        all_k = apply_rotary(all_k, cos[:sl], sin[:sl])

        all_q = all_q.permute(1, 2, 0, 3)
        all_k = all_k.permute(1, 2, 0, 3)
        all_v = all_v.permute(1, 2, 0, 3)

        output = torch.nn.functional.scaled_dot_product_attention(
            all_q, all_k, all_v, dropout_p=self.config.dropout if self.training else 0,
            is_causal=True)
        output = output.permute(0, 2, 1, 3).reshape(bs, sl, hs)
        return self.resid_dropout(self.o_proj(output))


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
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.LongTensor):
        # [batch size, seq length]
        assert input_ids.dim() == 2
        hidden_states = self.word_embedding_table(input_ids)
        hidden_states = self.dropout(hidden_states)
        layers_output = [hidden_states.detach()]
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            layers_output.append(hidden_states.detach())
        return self.rms(hidden_states), layers_output

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


class CausalLlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.model = Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config = config

        # init all weights
        self.apply(self._init_weights)

    def forward(self, input_ids: torch.LongTensor, labels: Optional[torch.LongTensor] = None):
        hidden_states, _ = self.model(input_ids)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return logits, loss

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            block_size = self.config.max_context_length
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def test_modeling():
    ref_model_id = "felixdae/Llama-2-7b-hf"
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_id)

    config = LlamaConfig(num_hidden_layers=2)
    model = Model(config)
    model.load_weights_from_hf(ref_model_id)
    model.eval()

    input_ids = torch.LongTensor([[42, 2, 23], [5, 6, 9]])
    out1 = ref_model(input_ids, output_hidden_states=True)
    out2, layer_output = model(input_ids)

    delta = torch.abs(torch.max(out1.hidden_states[-1] - out2))
    assert delta < 1e-4, f"fail at final output, delta {delta}"

    for i in range(config.num_hidden_layers):
        t1 = out1.hidden_states[i]
        t2 = layer_output[i]
        delta = torch.abs(torch.max(t2 - t1))
        assert delta < 1e-4, f"fail at layer {i}, delta {delta}"
