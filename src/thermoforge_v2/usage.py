"""Codex 用量分段累计：0.153.4 恢复会话后，total 从新进程重新计数。"""

from copy import deepcopy
import uuid


def begin_process(track: dict) -> dict:
    """新进程开始前固定前段累计值，不清除已经发生的消费。"""
    return {"usage_generation": uuid.uuid4().hex,
            "usage_base": deepcopy((track.get("usage") or {}).get("total") or {}),
            "usage_process": {}}


def merge_usage(track: dict, raw: dict, generation: str) -> dict | None:
    """同进程只取累计计数最大值；通知和 turn 结果重复上报不会二次扣账。"""
    if generation != track.get("usage_generation"):
        return None  # 已退休进程的迟到通知不能污染恢复后的分段。
    incoming = {k: v for k, v in (raw.get("total") or {}).items()
                if isinstance(v, int) and not isinstance(v, bool) and v >= 0}
    if not incoming:
        return None
    if "totalTokens" not in incoming and {"inputTokens", "outputTokens"} <= incoming.keys():
        incoming["totalTokens"] = incoming["inputTokens"] + incoming["outputTokens"]
    previous = track.get("usage_process") or {}
    current = {k: max(previous.get(k, 0), incoming.get(k, 0)) for k in previous.keys() | incoming.keys()}
    base = track.get("usage_base") or {}
    cumulative = {k: base.get(k, 0) + current.get(k, 0) for k in base.keys() | current.keys()}
    usage = {k: deepcopy(v) for k, v in raw.items() if k != "total"}
    usage["total"] = cumulative
    return {"usage": usage, "usage_process": current}
