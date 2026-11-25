"""Explicit Qwen-family boundaries for source-native visual communication.

Keep the historical Qwen2.5 path unchanged. Qwen3.8's published checkpoint uses
the qwen3_5 implementation and a different decoder, vision return and M-RoPE API.
"""
from contextlib import contextmanager, nullcontext


SUPPORTED = ('qwen2_5_vl', 'qwen3_5')


def family(model):
    value = getattr(getattr(model, 'config', None), 'model_type', 'qwen2_5_vl')
    if value not in SUPPORTED:
        raise ValueError('unsupported native-visual backbone: ' + str(value))
    return value


def text_config(model):
    return model.config.text_config if family(model) == 'qwen3_5' else model.config


def visual_module(model):
    return model.model.visual if family(model) == 'qwen3_5' else model.visual


def decoder_layers(model):
    return model.model.language_model.layers if family(model) == 'qwen3_5' else model.model.layers


def visual_merger_output(model, pixels, grid):
    result = visual_module(model)(pixels, grid_thw=grid)
    if family(model) == 'qwen3_5':
        # pooler_output contains the actual post-merger decoder-compatible rows.
        # last_hidden_state is a different representation and must not be used.
        result = result.pooler_output
    return result


def lora_targets(model):
    if family(model) == 'qwen2_5_vl':
        return ['q_proj', 'v_proj']
    import torch
    names = {'q_proj', 'v_proj', 'in_proj_qkv', 'in_proj_qkvz', 'in_proj_z',
             'in_proj_a', 'in_proj_b', 'out_proj'}
    result = [name for name, module in model.named_modules()
        if 'language_model.layers.' in name and name.rsplit('.', 1)[-1] in names
        and isinstance(module, torch.nn.Linear)]
    if not result or not any('.linear_attn.' in name for name in result) or not any('.self_attn.' in name for name in result):
        raise ValueError('hybrid decoder requires adapters in both attention families')
    return result


def native_inputs(model, ids, embeds, grids, *, for_generation=False):
    """Prepare one real decoder consumption without invoking the visual tower.

    Qwen3.5 forward accepts exactly one of input_ids/inputs_embeds. Its generation
    entry also needs prompt IDs for output-prefix bookkeeping; Transformers drops
    those IDs from the first forward when supplied with inputs_embeds.
    """
    import torch
    if family(model) == 'qwen2_5_vl':
        args = {'input_ids': ids, 'inputs_embeds': embeds, 'attention_mask': torch.ones_like(ids)}
        if grids is not None:
            args['image_grid_thw'] = grids
        model.rope_deltas = None
        return args
    owner = model.model
    owner.rope_deltas = None
    modality = (ids == model.config.image_token_id).to(torch.int32)
    positions, deltas = owner.get_rope_index(ids, modality, image_grid_thw=grids,
        attention_mask=torch.ones_like(ids))
    if positions.shape != (3, *ids.shape):
        raise ValueError('unexpected multimodal position shape')
    owner.rope_deltas = deltas
    args = {'inputs_embeds': embeds, 'attention_mask': torch.ones_like(ids),
            'position_ids': positions}
    if for_generation:
        args.update(input_ids=ids, mm_token_type_ids=modality)
        if grids is not None:
            args['image_grid_thw'] = grids
    return args


def cache_options(model, device, sampling):
    if family(model) == 'qwen3_5':
        # Its recurrent state is not a Qwen2.5 DynamicCache/OffloadedCache.
        return {}
    from .qwen_policy import generation_cache_options
    return generation_cache_options(device, sampling)


@contextmanager
def generation_context(model, device, sampling):
    import torch
    from .qwen_policy import generation_last_position, generation_chunked_mlp
    from .generation_attention import generation_bounded_attention
    from .native_prefill_compat import native_embedding_chunked_prefill
    with generation_last_position(model.lm_head), generation_chunked_mlp(
            [layer.mlp for layer in decoder_layers(model)]), generation_bounded_attention(
            model, enabled=family(model) == 'qwen2_5_vl' and not sampling and torch.device(device).type == 'cuda'), (
            native_embedding_chunked_prefill(model) if family(model) == 'qwen3_5' else nullcontext()):
        yield
