"""Bound generation attention-mask storage while retaining every allowed key."""
from contextlib import contextmanager
from dataclasses import dataclass, replace
from types import MethodType
import torch


@dataclass
class DeferredCausalMask:
    positions: torch.Tensor
    padding: torch.Tensor | None
    window: int | None
    key_length: int | None = None

    def __getitem__(self, index):
        # The installed Qwen SDPA implementation slices only the key dimension.
        if len(index) != 4 or any(x != slice(None) for x in index[:3]):
            raise ValueError('unsupported deferred mask slice')
        return replace(self, key_length=index[3].stop)


def bounded_causal_sdpa(query, key, value, mask, chunk_size=256):
    """Query chunks share full K/V; reproduce Qwen's additive causal/window mask."""
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('positive query chunk required')
    if query.shape[-2] != mask.positions.numel():
        raise ValueError('query/cache-position mismatch')
    if mask.key_length is not None and mask.key_length != key.shape[-2]:
        raise ValueError('key slice mismatch')
    output = torch.empty_like(query)
    columns = torch.arange(key.shape[-2], device=query.device)
    minimum = torch.finfo(query.dtype).min
    for start in range(0, query.shape[-2], chunk_size):
        end = min(start+chunk_size, query.shape[-2])
        positions = mask.positions[start:end].to(query.device).reshape(-1, 1)
        blocked = columns > positions
        if mask.window is not None:
            blocked |= columns <= positions-mask.window
        additive = torch.zeros((query.shape[0], 1, end-start, key.shape[-2]),
                               dtype=query.dtype, device=query.device)
        additive.masked_fill_(blocked[None, None], minimum)
        if mask.padding is not None:
            padding = mask.padding[:, :key.shape[-2]].to(query.device)
            additive.masked_fill_(padding[:, None, None, :] == 0, minimum)
        output[:, :, start:end] = torch.nn.functional.scaled_dot_product_attention(
            query[:, :, start:end].contiguous(), key, value, attn_mask=additive,
            dropout_p=0.0, is_causal=False)
    return output


@contextmanager
def generation_bounded_attention(model, enabled=True, threshold=8192, chunk_size=256):
    """Generation-only instance overrides; fallback retains native training/masks.

    This adapter targets the installed Qwen2.5-VL SDPA interface. It preserves
    the library's sliding_window behavior even when use_sliding_window is false.
    Only unpadded DynamicCache/OffloadedCache generation is intercepted.
    """
    if not enabled:
        yield
        return
    from transformers.cache_utils import DynamicCache, OffloadedCache
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLSdpaAttention, apply_multimodal_rotary_pos_emb, repeat_kv)
    decoder = model.model
    original_mask = decoder._update_causal_mask
    originals = []

    def update(this, attention_mask, input_tensor, cache_position, past_key_values,
               output_attentions=False):
        key_length = input_tensor.shape[1]+(past_key_values.get_seq_length()
                                           if past_key_values is not None else 0)
        native = (this.training or torch.is_grad_enabled() or output_attentions or
                  this.config._attn_implementation != 'sdpa' or key_length < threshold or
                  (past_key_values is not None and not isinstance(past_key_values, (DynamicCache, OffloadedCache))) or
                  (attention_mask is not None and (attention_mask.ndim != 2 or not bool(torch.all(attention_mask == 1)))))
        if native:
            return original_mask(attention_mask, input_tensor, cache_position,
                                 past_key_values, output_attentions)
        return DeferredCausalMask(cache_position, attention_mask, this.config.sliding_window)

    def forward(this, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, _original=None):
        if not isinstance(attention_mask, DeferredCausalMask):
            return _original(hidden_states, attention_mask, position_ids, past_key_value,
                             output_attentions, use_cache, cache_position, position_embeddings)
        batch, length, _ = hidden_states.shape
        q = this.q_proj(hidden_states).view(batch, length, -1, this.head_dim).transpose(1, 2)
        k = this.k_proj(hidden_states).view(batch, length, -1, this.head_dim).transpose(1, 2)
        v = this.v_proj(hidden_states).view(batch, length, -1, this.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, this.rope_scaling['mrope_section'])
        if past_key_value is not None:
            k, v = past_key_value.update(k, v, this.layer_idx,
                {'sin': sin, 'cos': cos, 'cache_position': cache_position})
        k = repeat_kv(k, this.num_key_value_groups).contiguous()
        v = repeat_kv(v, this.num_key_value_groups).contiguous()
        attended = bounded_causal_sdpa(q, k, v, attention_mask, chunk_size)
        attended = attended.transpose(1, 2).contiguous().view(batch, length, this.hidden_size)
        return this.o_proj(attended), None, past_key_value

    try:
        if not all(isinstance(layer.self_attn, Qwen2_5_VLSdpaAttention) for layer in decoder.layers):
            raise ValueError('bounded generation requires Qwen2.5-VL SDPA layers')
        decoder._update_causal_mask = MethodType(update, decoder)
        for layer in decoder.layers:
            attention = layer.self_attn
            original = attention.forward
            originals.append((attention, original))
            def adapted(this, *args, _original=original, **kwargs):
                return forward(this, *args, _original=_original, **kwargs)
            attention.forward = MethodType(adapted, attention)
        yield
    finally:
        decoder._update_causal_mask = original_mask
        for attention, original in originals:
            attention.forward = original
