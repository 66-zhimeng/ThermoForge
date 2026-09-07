"""Identity of an exact experiment request, independent of its research narrative.

The caller supplies the frozen protocol fingerprint, which covers the dataset,
features, target, split, seed, runtime environment and evaluation settings.
Canonical JSON removes dictionary ordering and omitted ModelSpec defaults only;
it does not infer equivalent algorithms, parameters or source code.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from thermoforge_core.contracts.experiment import ModelSpec


def experiment_fingerprint(protocol_fingerprint: str,
                           model_spec: Mapping[str, Any],
                           lab_content_hash: str | None = None) -> str:
    """Return a stable SHA-256 for a validated, frozen experiment request.

    Lab models require their immutable source hash in addition to their registry
    reference. The reference stays in the identity because the current execution
    contract passes ``hyperparameters.lab`` to user code: aliases can therefore
    affect behavior. Matching source alone is insufficient to authorize reuse.
    Numeric types and all explicit parameters are likewise preserved. This is
    deliberately conservative; semantic equivalence is a scientific judgement.
    """
    if not isinstance(protocol_fingerprint, str) or not protocol_fingerprint.strip():
        raise ValueError("实验指纹必须包含冻结评价协议的 fingerprint")
    if not isinstance(model_spec, Mapping):
        raise ValueError("model_spec 必须为模型规格对象")
    model = ModelSpec.model_validate(dict(model_spec)).model_dump(mode="python")
    if model["category"] == "lab":
        if not isinstance(lab_content_hash, str) or not re.fullmatch(
                r"[0-9a-fA-F]{64}", lab_content_hash):
            raise ValueError("实验室模型必须提供不可变源码的 SHA-256 content_hash")
        source_hash = lab_content_hash.lower()
    elif lab_content_hash is not None:
        raise ValueError("内置模型不能附带实验室源码 content_hash")
    else:
        source_hash = None
    identity = {"schema": "thermoforge-experiment-v1",
                "protocol_fingerprint": protocol_fingerprint,
                "model": model, "lab_content_hash": source_hash}
    try:
        serialized = json.dumps(identity, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("实验身份字段必须为有限、可序列化的 JSON 值") from exc
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
