import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import CrossEntropyLoss


@dataclass
class RwkvConfig:
    bos_token_id: int = 0
    eos_token_id: int = 0
    hidden_size: int = 768
    attention_hidden_size: int = 768
    intermediate_size: int = 3072
    context_length: int = 1024
    layer_norm_epsilon: float = 1e-05
    num_hidden_layers: int = 12
    rescale_every: int = 6
    vocab_size: int = 50277


class Mixer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.mix_weight = nn.Parameter(torch.empty(1, 1, hidden_size))

    def forward(self, current: torch.Tensor, previous: torch.Tensor):
        # [batch size, seq length, hidden size]
        assert len(current.size()) == 3

        mixed = self.mix_weight * current + (1 - self.mix_weight) * previous
        return mixed


class FeedForward(nn.Module):
    def __init__(self, config: RwkvConfig, layer_id):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.key = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.receptance = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.value = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        self.key_mixer = Mixer(config.hidden_size)
        self.receptance_mixer = Mixer(config.hidden_size)

        self.config = config
        self.layer_id = layer_id

    def forward(self, hidden_states: torch.Tensor):
        current = hidden_states
        # shift along time dimension
        previous = self.time_shift(current)

        key = self.key(self.key_mixer(current, previous))
        key = torch.square(torch.relu(key))
        value = self.value(key)

        receptance = torch.sigmoid(self.receptance(self.receptance_mixer(current, previous)))
        return receptance * value


class Memory(nn.Module):
    def __init__(self, config: RwkvConfig):
        super().__init__()
        self.time_decay = nn.Parameter(torch.empty(config.attention_hidden_size))
        self.time_first = nn.Parameter(torch.empty(config.attention_hidden_size))

    def forward(self, key: torch.Tensor, value: torch.Tensor):
        device = key.device
        batch_size, seq_length, hidden_size = key.size()
        u = self.time_first
        w = -torch.exp(self.time_decay)

        lst = []
        a = torch.zeros(batch_size, hidden_size).to(device)
        b = torch.zeros_like(a).to(device)
        exponent = torch.full_like(a, -float('inf')).to(device)
        for t in range(seq_length):
            kt = key[:, t]

            # compute wkv
            max_exponent = torch.max(exponent, u + kt)
            wt = torch.exp(u + kt - max_exponent)
            vt = value[:, t]
            scale = torch.exp(exponent - max_exponent)
            wkv = (a * scale + wt * vt) / (b * scale + wt)

            # update state
            max_exponent = torch.max(exponent + w, kt)
            scale1 = torch.exp(exponent + w - max_exponent)
            scale2 = torch.exp(kt - max_exponent)
            a = scale1 * a + scale2 * vt
            b = scale1 * b + scale2
            exponent = max_exponent

            lst.append(wkv.unsqueeze(1))
        return torch.concat(lst, dim=1)


class Attention(nn.Module):
    def __init__(self, config: RwkvConfig, layer_id):
        super().__init__()
        attention_hidden_size = config.attention_hidden_size
        hidden_size = config.hidden_size

        self.key_mixer = Mixer(config.hidden_size)
        self.value_mixer = Mixer(config.hidden_size)
        self.receptance_mixer = Mixer(config.hidden_size)

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.key = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        self.receptance = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        self.output = nn.Linear(attention_hidden_size, hidden_size, bias=False)

        self.layer_id = layer_id
        self.config = config

        self.memory = Memory(config)

    def forward(self, hidden_states: torch.Tensor):
        assert len(hidden_states.size()) == 3
        current = hidden_states
        # shift along time dimension
        previous = self.time_shift(current)

        key = self.key(self.key_mixer(current, previous))
        value = self.value(self.value_mixer(current, previous))
        receptance = torch.sigmoid(self.receptance(self.receptance_mixer(current, previous)))

        rwkv = self.memory(key, value)
        output = self.output(receptance * rwkv)
        return output


class Block(nn.Module):
    def __init__(self, config: RwkvConfig, layer_id):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attention = Attention(config, layer_id)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ffn = FeedForward(config, layer_id)

    def forward(self, hidden_states: torch.Tensor):
        attention = self.attention(self.ln1(hidden_states))
        hidden_states = hidden_states + attention

        feed_forward = self.ffn(self.ln2(hidden_states))
        hidden_states = hidden_states + feed_forward
        return hidden_states


def should_rescale(idx: int, rescale_every: int, layers_are_rescaled: bool):
    return layers_are_rescaled and rescale_every > 0 and (idx + 1) % rescale_every == 0


class Model(nn.Module):
    def __init__(self, config: RwkvConfig):
        super().__init__()
        self.config = config
        self.layers_are_rescaled = False
        self.word_embedding_table = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Block(config, layer_id) for layer_id in range(config.num_hidden_layers)])
        self.pre_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.post_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, input_ids: torch.LongTensor):
        """
        :param input_ids: shape [batch_size, seq_length]
        :return:
        """
        assert len(input_ids.size()) == 2
        if self.config.rescale_every > 0:
            self._try_rescale_layers()

        hidden_states = self.word_embedding_table(input_ids)
        hidden_states = self.pre_ln(hidden_states)
        for idx, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states)
            if should_rescale(idx, self.config.rescale_every, self.layers_are_rescaled):
                hidden_states = hidden_states / 2
        last = self.post_ln(hidden_states)
        return last

    def _try_rescale_layers(self):
        # inference
        if not self.training:
            if self.layers_are_rescaled:
                return
            # rescale
            with torch.no_grad():
                for block_id, block in enumerate(self.layers):
                    block.attention.output.weight.div_(2 ** int(block_id // self.config.rescale_every))
                    block.ffn.value.weight.div_(2 ** int(block_id // self.config.rescale_every))
            self.layers_are_rescaled = not self.layers_are_rescaled
            return

        if not self.layers_are_rescaled:
            return
        # revert
        with torch.no_grad():
            for block_id, block in enumerate(self.layers):
                block.attention.output.weight.mul_(2 ** int(block_id // self.config.rescale_every))
                block.ffn.value.weight.mul_(2 ** int(block_id // self.config.rescale_every))
        self.layers_are_rescaled = not self.layers_are_rescaled


def name_mapping(param: str):
    out = {
        "word_embedding_table.weight": "rwkv.embeddings.weight",
        "pre_ln.weight": "rwkv.blocks.0.pre_ln.weight",
        "pre_ln.bias": "rwkv.blocks.0.pre_ln.bias",
        "post_ln.weight": "rwkv.ln_out.weight",
        "post_ln.bias": "rwkv.ln_out.bias",
    }
    if param in out:
        return out[param]

    arr = param.split('.')
    assert arr[0] == 'layers'
    layer_id = arr[1]
    sub = arr[2]

    prefix = f"rwkv.blocks.{layer_id}"
    if sub in ['ln1', 'ln2']:
        return prefix + "." + sub + "." + arr[-1]
    if sub == 'attention':
        if 'time_decay' in param or 'time_first' in param:
            return prefix + "." + sub + "." + arr[-1]
        if 'key_mixer' in param:
            return prefix + "." + sub + ".time_mix_key"
        if 'value_mixer' in param:
            return prefix + "." + sub + ".time_mix_value"
        if 'receptance_mixer' in param:
            return prefix + "." + sub + ".time_mix_receptance"
        return prefix + "." + sub + f".{arr[-2]}.{arr[-1]}"
    if sub == 'ffn':
        if 'key_mixer' in param:
            return prefix + ".feed_forward.time_mix_key"
        if 'receptance_mixer' in param:
            return prefix + ".feed_forward.time_mix_receptance"
        return prefix + f".feed_forward.{arr[-2]}.{arr[-1]}"


class CausalRwkvModel(nn.Module):
    def __init__(self, config: RwkvConfig):
        super().__init__()
        self.rwkv = Model(config)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config = config

        # init all weights
        self.apply(self._init_weights)

    def forward(self, input_ids: torch.LongTensor, labels: Optional[torch.LongTensor] = None):
        hidden_states = self.rwkv(input_ids)
        logits = self.head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return logits, loss

    # copy from hugging face rwkv implementation
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, Attention):
            layer_id = module.layer_id
            num_hidden_layers = module.config.num_hidden_layers
            hidden_size = module.config.hidden_size
            attention_hidden_size = module.config.attention_hidden_size

            ratio_0_to_1 = layer_id / (num_hidden_layers - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / num_hidden_layers)  # 1 to ~0

            time_weight = torch.tensor(
                [i / hidden_size for i in range(hidden_size)],
                dtype=module.key_mixer.mix_weight.dtype,
                device=module.key_mixer.mix_weight.device,
            )
            time_weight = time_weight[None, None, :]

            decay_speed = [
                -5 + 8 * (h / (attention_hidden_size - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
                for h in range(attention_hidden_size)
            ]
            decay_speed = torch.tensor(decay_speed, dtype=module.memory.time_decay.dtype,
                                       device=module.memory.time_decay.device)
            zigzag = (
                    torch.tensor(
                        [(i + 1) % 3 - 1 for i in range(attention_hidden_size)],
                        dtype=module.memory.time_first.dtype,
                        device=module.memory.time_first.device,
                    )
                    * 0.5
            )

            with torch.no_grad():
                module.memory.time_decay.data = decay_speed
                module.memory.time_first.data = torch.ones_like(module.memory.time_first * math.log(0.3) + zigzag)

                module.key_mixer.mix_weight.data = torch.pow(time_weight, ratio_1_to_almost0)
                module.value_mixer.mix_weight.data = torch.pow(time_weight, ratio_1_to_almost0) + 0.3 * ratio_0_to_1
                module.receptance_mixer.mix_weight.data = torch.pow(time_weight, 0.5 * ratio_1_to_almost0)
        elif isinstance(module, FeedForward):
            layer_id = module.layer_id
            num_hidden_layers = module.config.num_hidden_layers
            hidden_size = module.config.hidden_size

            ratio_1_to_almost0 = 1.0 - (layer_id / num_hidden_layers)  # 1 to ~0

            time_weight = torch.tensor(
                [i / hidden_size for i in range(hidden_size)],
                dtype=module.key_mixer.mix_weight.dtype,
                device=module.key_mixer.mix_weight.device,
            )
            time_weight = time_weight[None, None, :]

            with torch.no_grad():
                module.key_mixer.mix_weight.data = torch.pow(time_weight, ratio_1_to_almost0)
                module.receptance_mixer.mix_weight.data = torch.pow(time_weight, ratio_1_to_almost0)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            block_size = self.config.context_length
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
