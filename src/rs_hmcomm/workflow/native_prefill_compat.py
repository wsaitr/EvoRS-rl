"""Preserve native embeddings and M-RoPE in Transformers 5.17 chunked prefill.

Keep the library's actual chunk loop/cache implementation. Its 5.17 loop omits
the first-iteration flag and slices positions as 2-D, so enabling its option
alone discards supplied visual embeddings. Adapt only each chunk's inputs.
Upstream: transformers/generation/utils.py, GenerationMixin._prefill (Apache-2.0).
"""
from contextlib import contextmanager
from types import MethodType


def inference_prefill_options(input_tokens):
    """Keep the existing short-context computation; bound only long prefills."""
    if type(input_tokens) is not int or input_tokens < 1:
        raise ValueError('A positive actual token count is required')
    return {'prefill_chunk_size': 2048} if input_tokens >= 16384 else {}


def _restore(instance, name, previous):
    present, value = previous
    if present:
        setattr(instance, name, value)
    else:
        delattr(instance, name)


@contextmanager
def native_embedding_chunked_prefill(model):
    original_prefill = model._prefill
    previous_prefill = ('_prefill' in model.__dict__, model.__dict__.get('_prefill'))

    def prefill(this, input_ids, generation_config, model_kwargs, is_first_iteration=True):
        embeds = model_kwargs.get('inputs_embeds')
        if generation_config.prefill_chunk_size is None or embeds is None:
            return original_prefill(input_ids, generation_config, model_kwargs,
                                    is_first_iteration=is_first_iteration)
        cache = model_kwargs.get('past_key_values')
        if cache is None or int(cache.get_seq_length()) != 0:
            raise ValueError('Native chunked prefill requires a fresh actual cache')
        if embeds.shape[:2] != input_ids.shape:
            raise ValueError('Native prefill IDs and embedding positions differ')
        positions = model_kwargs.pop('position_ids', None)
        if positions is not None and positions.shape[-1] != input_ids.shape[-1]:
            raise ValueError('Native prefill M-RoPE positions differ')
        model_kwargs.pop('inputs_embeds')
        original_prepare = this.prepare_inputs_for_generation
        previous_prepare = ('prepare_inputs_for_generation' in this.__dict__,
                            this.__dict__.get('prepare_inputs_for_generation'))

        def prepare(owner, chunk, **kwargs):
            start = int(kwargs['past_key_values'].get_seq_length())
            end = start + chunk.shape[-1]
            if not 0 <= start < end <= embeds.shape[1]:
                raise ValueError('Library chunk positions exceed the native prefill')
            for name in ('inputs_embeds', 'position_ids', 'is_first_iteration'):
                kwargs.pop(name, None)
            # These are per-token metadata, unlike the full immutable image grid.
            for name in ('token_type_ids', 'mm_token_type_ids'):
                if kwargs.get(name) is not None:
                    kwargs[name] = kwargs[name][..., start:end]
            return original_prepare(chunk, inputs_embeds=embeds[:, start:end],
                position_ids=None if positions is None else positions[..., start:end],
                is_first_iteration=True, **kwargs)

        this.prepare_inputs_for_generation = MethodType(prepare, this)
        try:
            return original_prefill(input_ids, generation_config, model_kwargs,
                                    is_first_iteration=is_first_iteration)
        finally:
            _restore(this, 'prepare_inputs_for_generation', previous_prepare)
            model_kwargs['inputs_embeds'] = embeds
            model_kwargs['position_ids'] = positions

    model._prefill = MethodType(prefill, model)
    try:
        yield
    finally:
        _restore(model, '_prefill', previous_prefill)
