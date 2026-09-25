"""Small validation helpers shared by EvoRS contract records.

The contract layer intentionally accepts only JSON-compatible metadata.  Heavy
objects such as PIL images, tensors, numpy arrays and byte buffers must be kept
outside these records and addressed through string references.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real
from types import MappingProxyType
from typing import Any, TypeVar


EnumT = TypeVar("EnumT")


def require_id(value: str, field_name: str) -> str:
    """Validate an opaque record identifier or record reference."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value or not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    if any(char.isspace() for char in value):
        raise ValueError(f"{field_name} must not contain whitespace")
    return value


def require_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


def require_step(value: int, field_name: str = "created_step") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def normalize_confidence(value: float | None, field_name: str = "confidence") -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number or None")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{field_name} must be finite and within [0, 1]")
    return normalized


def normalize_id_tuple(
    values: tuple[str, ...] | list[str],
    field_name: str,
    *,
    require_nonempty: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be a tuple or list of string references")
    normalized = tuple(require_id(value, f"{field_name}[{index}]") for index, value in enumerate(values))
    if require_nonempty and not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicate references")
    return normalized


def normalize_text_tuple(
    values: tuple[str, ...] | list[str],
    field_name: str,
) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be a tuple or list of strings")
    normalized = tuple(require_text(value, f"{field_name}[{index}]") for index, value in enumerate(values))
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicates")
    return normalized


def _validate_json_value(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            _validate_json_value(nested, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _validate_json_value(nested, f"{path}[{index}]")
        return
    raise TypeError(
        f"{path} contains unsupported value of type {type(value).__name__}; "
        "store images, tensors and binary data externally and use a string ref"
    )


def _freeze_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json_value(nested) for key, nested in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(nested) for nested in value)
    return value


def _thaw_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(nested) for nested in value]
    return value


def normalize_json_object(value: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    _validate_json_value(value, field_name)
    return _freeze_json_value(value)


def json_object_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("value must be a mapping")
    return _thaw_json_value(value)


def strict_dict(
    data: Mapping[str, Any],
    type_name: str,
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise TypeError(f"{type_name}.from_dict expects a mapping")
    optional = optional or set()
    keys = set(data)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise ValueError(f"{type_name} is missing required fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{type_name} contains unknown fields: {sorted(unknown)}")
    return dict(data)


def coerce_enum(value: Any, enum_type: type[EnumT], field_name: str) -> EnumT:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = [item.value for item in enum_type]  # type: ignore[attr-defined]
        raise ValueError(f"{field_name} must be one of {allowed}") from exc
