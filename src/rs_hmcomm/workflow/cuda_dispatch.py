"""Validate explicitly supplied CUDA-only inference dispatch without moving it."""


def _cuda_name(value):
    if type(value) is int and value >= 0:
        return 'cuda:'+str(value)
    value=str(value)
    if value.startswith('cuda:') and value[5:].isdigit():
        return 'cuda:'+str(int(value[5:]))
    raise ValueError('inference dispatch must use explicit CUDA devices only')


def validate_cuda_dispatch(device_map, *, entry_device, embedding_device, visual_devices):
    if not isinstance(device_map,dict) or not device_map:
        raise ValueError('a loaded explicit device map is required')
    normalized={name:_cuda_name(value) for name,value in device_map.items()}
    entry=_cuda_name(entry_device)
    if len(set(normalized.values())) < 2:
        raise ValueError('sharded inference requires multiple CUDA devices')
    if _cuda_name(embedding_device)!=entry:
        raise ValueError('input embeddings must remain on the entry device')
    if not visual_devices or {_cuda_name(value) for value in visual_devices}!={entry}:
        raise ValueError('native visual encoding must remain on the entry device')
    if entry not in normalized.values():
        raise ValueError('entry device is absent from dispatch')
    return normalized
