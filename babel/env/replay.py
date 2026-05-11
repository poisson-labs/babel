from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import msgpack
import numpy as np

JsonLike = dict[str, Any] | list[Any] | str | int | float | bool | None
_msgpack = cast(Any, msgpack)


@dataclass(slots=True)
class ReplayRecorder:
    seed: int
    config: dict[str, JsonLike]
    path: Path | None = None
    per_step: list[dict[str, JsonLike]] = field(default_factory=lambda: [])
    outcomes: dict[str, JsonLike] = field(default_factory=lambda: {})

    def record_step(
        self,
        *,
        step: int,
        env_state: dict[str, Any],
        per_agent: list[dict[str, Any]],
    ) -> None:
        self.per_step.append(
            {
                "step": step,
                "env_state": _safe_dict(env_state),
                "per_agent": _safe_list(per_agent),
            }
        )

    def finish(self, outcomes: dict[str, JsonLike]) -> dict[str, JsonLike]:
        self.outcomes = _safe_dict(outcomes)
        replay = self.to_dict()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("wb") as handle:
                handle.write(cast(bytes, _msgpack.packb(replay, use_bin_type=True)))
        return replay

    def to_dict(self) -> dict[str, JsonLike]:
        return {
            "seed": self.seed,
            "config": _safe_dict(self.config),
            "per_step": _safe_list(self.per_step),
            "outcomes": _safe_dict(self.outcomes),
        }


def load_replay(path: str | Path) -> dict[str, JsonLike]:
    with Path(path).open("rb") as handle:
        replay = _msgpack.unpackb(handle.read(), raw=False)
    if not isinstance(replay, dict):
        raise ValueError(f"Replay at {path} did not unpack to a mapping")
    return cast(dict[str, JsonLike], replay)


def _safe_dict(value: dict[str, Any]) -> dict[str, JsonLike]:
    converted = _to_msgpack_safe(value)
    if not isinstance(converted, dict):
        raise TypeError("Expected msgpack-safe dict")
    return cast(dict[str, JsonLike], converted)


def _safe_list(value: list[Any]) -> list[JsonLike]:
    converted = _to_msgpack_safe(value)
    if not isinstance(converted, list):
        raise TypeError("Expected msgpack-safe list")
    return cast(list[JsonLike], converted)


def _to_msgpack_safe(value: Any) -> JsonLike:
    if isinstance(value, np.ndarray):
        return _to_msgpack_safe(value.tolist())
    if isinstance(value, np.generic):
        return _to_msgpack_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_to_msgpack_safe(item) for item in cast(tuple[object, ...], value)]
    if isinstance(value, list):
        return [_to_msgpack_safe(item) for item in cast(list[object], value)]
    if isinstance(value, dict):
        return {
            str(key): _to_msgpack_safe(item)
            for key, item in cast(dict[object, object], value).items()
        }
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    raise TypeError(f"Unsupported replay value type: {type(value)!r}")
