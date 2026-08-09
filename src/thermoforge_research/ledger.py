"""Research Ledger（research-loop.md §8、conventions.md §1.3/§7.9）。

目录结构::

    research/
    ├── goals/RG-0001.yaml
    ├── hypotheses/H-0001.yaml
    ├── experiments/EXP-0001/experiment.yaml   # + 运行制品（runner 写入）
    ├── findings/F-0001.md                     # YAML front matter + 正文
    ├── decisions/D-0001.yaml
    ├── models/M-0001.yaml
    ├── views/VIEW-0001.yaml                   # Dataset View 定义（runner 引用）
    ├── ids/                                   # 顺序 ID 分配器（core.ids）
    └── ledger_index.json                      # JSON 索引（单写者，原子替换）

- ID 用 `thermoforge_core.ids.IdAllocator` 分配，不复用、不回收。
- 每次状态转换记录原因、输入/输出制品、执行者、时间和错误信息
  （research-loop §2）。
- 实验状态 completed 后不可变：任何修改/再转换报 **TFX-904**。
- 单写者：文件锁互斥（implementation-notes §10.2）；索引为 JSON。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from thermoforge_core.ids import IdAllocator

from .errors import ResearchError

if sys.platform == "win32":
    import msvcrt

    def _lock(fp) -> None:
        fp.seek(0)
        msvcrt.locking(fp.fileno(), msvcrt.LK_LOCK, 1)

    def _unlock(fp) -> None:
        fp.seek(0)
        msvcrt.locking(fp.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _lock(fp) -> None:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)

    def _unlock(fp) -> None:
        fcntl.flock(fp.fileno(), fcntl.LOCK_UN)


_KIND_DIR = {
    "RG-": "goals",
    "H-": "hypotheses",
    "EXP-": "experiments",
    "F-": "findings",
    "D-": "decisions",
    "M-": "models",
    "VIEW-": "views",
}

_EXPERIMENT_STATUSES = ("created", "running", "completed", "failed")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class ResearchLedger:
    """研究账本（单写者）。

    用法::

        ledger = ResearchLedger(Path("research"))
        goal = ledger.create_goal("冷水机输入功率模型", actor="agent")
        hyp = ledger.create_hypothesis(goal["id"], "冷却水温度主导功率", actor="agent")
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        for sub in (*_KIND_DIR.values(), "ids"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self.allocator = IdAllocator(self.root / "ids")
        self._lock_path = self.root / "ledger.lock"
        self._index_path = self.root / "ledger_index.json"

    # ---------------------------------------------------------------- 内部

    def _locked(self):
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            with open(self._lock_path, "a+b") as fp:
                _lock(fp)
                try:
                    yield
                finally:
                    _unlock(fp)

        return _cm()

    def _read_index(self) -> dict[str, Any]:
        if not self._index_path.exists():
            return {"entities": {}}
        with open(self._index_path, encoding="utf-8") as fp:
            return json.load(fp)

    def _write_index_atomic(self, index: Mapping[str, Any]) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=".tmp_", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fp:
            json.dump(index, fp, ensure_ascii=False, sort_keys=True, indent=2)
            fp.write("\n")
        os.replace(tmp, self._index_path)

    def _entity_path(self, entity_id: str) -> Path:
        prefix = next(p for p in _KIND_DIR if entity_id.startswith(p))
        sub = self.root / _KIND_DIR[prefix]
        if prefix == "EXP-":
            return sub / entity_id / "experiment.yaml"
        if prefix == "F-":
            return sub / f"{entity_id}.md"
        return sub / f"{entity_id}.yaml"

    def _write_entity(self, entity: Mapping[str, Any]) -> None:
        path = self._entity_path(str(entity["id"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_",
                                   suffix=path.suffix)
        if path.suffix == ".md":
            body = str(entity.get("body") or "")
            doc = {k: v for k, v in entity.items() if k != "body"}
            text = ("---\n" + yaml.safe_dump(doc, allow_unicode=True,
                                             sort_keys=True)
                    + "---\n\n" + body + ("\n" if body else ""))
        else:
            text = yaml.safe_dump(dict(entity), allow_unicode=True,
                                  sort_keys=True)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fp:
            fp.write(text)
        os.replace(tmp, path)

    def _read_entity(self, entity_id: str) -> dict[str, Any]:
        path = self._entity_path(entity_id)
        if not path.exists():
            raise KeyError(f"账本实体不存在: {entity_id}")
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".md":
            parts = text.split("---\n", 2)
            doc = yaml.safe_load(parts[1])
            doc["body"] = parts[2].strip() if len(parts) > 2 else ""
            return doc
        return yaml.safe_load(text)

    @staticmethod
    def _index_entry(entity: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "kind": entity["kind"],
            "status": entity["status"],
            "refs": entity.get("refs", {}),
            "updated_at": entity["transitions"][-1]["at"],
        }

    def _store(self, entity: dict[str, Any]) -> dict[str, Any]:
        """在锁内写实体文件并更新索引。"""
        self._write_entity(entity)
        index = self._read_index()
        index["entities"][entity["id"]] = self._index_entry(entity)
        self._write_index_atomic(index)
        return entity

    def _new_entity(
        self,
        prefix: str,
        kind: str,
        status: str,
        fields: Mapping[str, Any],
        refs: Mapping[str, Any],
        *,
        actor: str,
        reason: str,
        entity_id: str | None = None,
    ) -> dict[str, Any]:
        eid = entity_id or self.allocator.allocate(prefix)
        if not eid.startswith(prefix):
            raise ValueError(f"实体 ID {eid!r} 与前缀 {prefix} 不符")
        if self._entity_path(eid).exists():
            raise ValueError(f"实体已存在: {eid}")
        entity = {
            "id": eid,
            "kind": kind,
            "status": status,
            "created_at": _utcnow(),
            "created_by": actor,
            **dict(fields),
            "refs": dict(refs),
            "transitions": [{
                "from": None,
                "to": status,
                "reason": reason or "创建",
                "inputs": [],
                "outputs": [],
                "actor": actor,
                "at": _utcnow(),
                "error": None,
            }],
        }
        return self._store(entity)

    def _require(self, entity_id: str, expected_prefix: str) -> dict[str, Any]:
        if not entity_id.startswith(expected_prefix):
            raise ValueError(f"{entity_id!r} 不是 {expected_prefix} 实体")
        path = self._entity_path(entity_id)
        if not path.exists():
            raise ValueError(f"引用的实体不存在: {entity_id}")
        return self._read_entity(entity_id)

    # ---------------------------------------------------------------- 创建

    def create_goal(self, name: str, *, actor: str, reason: str = "",
                    definition: Mapping[str, Any] | None = None,
                    entity_id: str | None = None) -> dict[str, Any]:
        """创建 Research Goal（RG-）。`entity_id` 可指定已分配的 ID
        （工具层先分配 ID 再校验契约定义，保证账本 ID 与契约 goal_id 一致）。"""
        with self._locked():
            return self._new_entity(
                "RG-", "goal", "active",
                {"name": name, "definition": dict(definition or {})},
                {}, actor=actor, reason=reason, entity_id=entity_id,
            )

    def create_hypothesis(
        self,
        goal_id: str,
        statement: str,
        *,
        actor: str,
        basis: Sequence[str] = (),
        reason: str = "",
    ) -> dict[str, Any]:
        """创建假设（H-）。`basis`：支撑证据（finding/experiment ID）。

        下一轮假设必须引用已有证据（research-loop §3）——`basis` 为空时
        仅允许目标下的首个假设。
        """
        with self._locked():
            self._require(goal_id, "RG-")
            for ref in basis:
                prefix = "F-" if ref.startswith("F-") else "EXP-"
                self._require(ref, prefix)
            if not basis:
                existing = [
                    e for e in self._read_index()["entities"].values()
                    if e["kind"] == "hypothesis"
                    and e["refs"].get("goal_id") == goal_id
                ]
                if existing:
                    raise ValueError(
                        "后续假设必须引用已有证据（research-loop §3），"
                        "basis 不能为空"
                    )
            return self._new_entity(
                "H-", "hypothesis", "unverified", {"statement": statement},
                {"goal_id": goal_id, "basis": list(basis)},
                actor=actor, reason=reason,
            )

    def register_view(self, definition: Mapping[str, Any], *,
                      actor: str, reason: str = "") -> dict[str, Any]:
        """登记 Dataset View 定义（VIEW-），供实验引用。"""
        with self._locked():
            return self._new_entity(
                "VIEW-", "view", "registered",
                {"definition": dict(definition)}, {}, actor=actor, reason=reason,
            )

    def register_experiment(
        self,
        definition: Mapping[str, Any],
        *,
        actor: str,
        reason: str = "",
    ) -> dict[str, Any]:
        """登记实验定义（EXP-）。定义中的 experiment_id 必须先由
        `allocate_id("EXP-")` 分配；goal / hypothesis / view 必须已存在。"""
        exp_id = str(definition.get("experiment_id", ""))
        goal_id = str(definition.get("goal_id", ""))
        hypothesis_id = str(definition.get("hypothesis_id", ""))
        view_id = str(definition.get("dataset_view", ""))
        with self._locked():
            self._require(goal_id, "RG-")
            self._require(hypothesis_id, "H-")
            self._require(view_id, "VIEW-")
            return self._new_entity(
                "EXP-", "experiment", "created",
                {"definition": dict(definition)},
                {"goal_id": goal_id, "hypothesis_id": hypothesis_id,
                 "dataset_view": view_id},
                actor=actor, reason=reason, entity_id=exp_id,
            )

    def create_finding(
        self,
        statement: str,
        *,
        actor: str,
        supported_by: Sequence[str] = (),
        hypothesis_id: str | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        """创建结构化发现（F-，Markdown + front matter）。

        `supported_by`：支持该结论的实验 ID（§8「哪些实验支持某个结论」）。
        """
        with self._locked():
            for exp_id in supported_by:
                self._require(exp_id, "EXP-")
            refs: dict[str, Any] = {"supported_by": list(supported_by)}
            if hypothesis_id:
                self._require(hypothesis_id, "H-")
                refs["hypothesis_id"] = hypothesis_id
            return self._new_entity(
                "F-", "finding", "proposed",
                {"statement": statement, "body": statement},
                refs, actor=actor, reason=reason,
            )

    def create_decision(
        self,
        subject: str,
        decision: str,
        *,
        actor: str,
        rationale: str = "",
        references: Sequence[str] = (),
        reason: str = "",
    ) -> dict[str, Any]:
        """创建决策记录（D-）。阈值修订等必须留痕（DD-13）。"""
        with self._locked():
            return self._new_entity(
                "D-", "decision", "active",
                {"subject": subject, "decision": decision,
                 "rationale": rationale},
                {"references": list(references)}, actor=actor, reason=reason,
            )

    def register_model(
        self,
        name: str,
        experiment_id: str,
        *,
        actor: str,
        metrics: Mapping[str, Any] | None = None,
        artifact_path: str | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        """登记模型（M-），状态 candidate；发布/拒绝经 transition 留痕。"""
        with self._locked():
            exp = self._require(experiment_id, "EXP-")
            return self._new_entity(
                "M-", "model", "candidate",
                {"name": name, "metrics": dict(metrics or {}),
                 "artifact_path": artifact_path},
                {"experiment_id": experiment_id,
                 "goal_id": exp["refs"].get("goal_id")},
                actor=actor, reason=reason,
            )

    # ---------------------------------------------------------------- 状态转换

    def transition(
        self,
        entity_id: str,
        to_status: str,
        *,
        reason: str,
        actor: str,
        inputs: Sequence[str] = (),
        outputs: Sequence[str] = (),
        error: str | None = None,
    ) -> dict[str, Any]:
        """状态转换（§2）：记录原因、输入/输出制品、执行者、时间、错误。

        实验 completed 后不可变：任何进一步转换报 TFX-904。
        """
        if not reason:
            raise ValueError("状态转换必须记录原因（research-loop §2）")
        with self._locked():
            entity = self._read_entity(entity_id)
            if entity["kind"] == "experiment":
                if entity["status"] == "completed":
                    raise ResearchError(
                        "TFX-904",
                        f"实验 {entity_id} 已完成，不可变（§7.9）",
                    )
                valid = set(_EXPERIMENT_STATUSES)
                if to_status not in valid:
                    raise ValueError(
                        f"非法实验状态: {to_status!r}（允许 {sorted(valid)}）"
                    )
            transition = {
                "from": entity["status"],
                "to": to_status,
                "reason": reason,
                "inputs": list(inputs),
                "outputs": list(outputs),
                "actor": actor,
                "at": _utcnow(),
                "error": error,
            }
            entity["transitions"].append(transition)
            entity["status"] = to_status
            return self._store(entity)

    # ---------------------------------------------------------------- 查询

    def get(self, entity_id: str) -> dict[str, Any]:
        return self._read_entity(entity_id)

    def transitions_of(self, entity_id: str) -> list[dict[str, Any]]:
        return list(self._read_entity(entity_id)["transitions"])

    def _entities(self, kind: str | None = None) -> list[dict[str, Any]]:
        index = self._read_index()
        ids = [
            eid for eid, e in index["entities"].items()
            if kind is None or e["kind"] == kind
        ]
        return [self._read_entity(eid) for eid in sorted(ids)]

    def experiments_supporting(self, finding_id: str) -> list[dict[str, Any]]:
        """哪些实验支持某个结论（§8）。"""
        finding = self._require(finding_id, "F-")
        return [self._read_entity(e)
                for e in finding["refs"].get("supported_by", [])]

    def list_goals(self) -> list[dict[str, Any]]:
        """列出全部 Research Goal 实体（工具层 tf_research_status 用）。"""
        return self._entities("goal")

    def unverified_hypotheses(self, goal_id: str | None = None) -> list[dict[str, Any]]:
        """哪些假设尚未验证（§8）。"""
        return [
            h for h in self._entities("hypothesis")
            if h["status"] == "unverified"
            and (goal_id is None or h["refs"].get("goal_id") == goal_id)
        ]

    def failed_experiments(self, goal_id: str | None = None) -> list[dict[str, Any]]:
        """哪些实验失败、失败原因（§8）。"""
        out = []
        for exp in self._entities("experiment"):
            if exp["status"] != "failed":
                continue
            if goal_id is not None and exp["refs"].get("goal_id") != goal_id:
                continue
            last = exp["transitions"][-1]
            out.append({
                "experiment_id": exp["id"],
                "failure_reason": last.get("error") or last.get("reason"),
                "failed_at": last["at"],
            })
        return out

    def goal_progress(self, goal_id: str) -> dict[str, Any]:
        """当前目标进展（§8）：各实体状态计数与最近活动时间。"""
        self._require(goal_id, "RG-")
        hypotheses = [h for h in self._entities("hypothesis")
                      if h["refs"].get("goal_id") == goal_id]
        experiments = [e for e in self._entities("experiment")
                       if e["refs"].get("goal_id") == goal_id]
        models = [m for m in self._entities("model")
                  if m["refs"].get("goal_id") == goal_id]
        by_status: dict[str, int] = {}
        for exp in experiments:
            by_status[exp["status"]] = by_status.get(exp["status"], 0) + 1
        latest = max(
            (e["transitions"][-1]["at"]
             for e in (*hypotheses, *experiments, *models)),
            default=None,
        )
        return {
            "goal_id": goal_id,
            "hypotheses": {
                "total": len(hypotheses),
                "unverified": sum(1 for h in hypotheses
                                  if h["status"] == "unverified"),
            },
            "experiments": {"total": len(experiments), "by_status": by_status},
            "models": {"total": len(models),
                       "candidates": sum(1 for m in models
                                         if m["status"] == "candidate")},
            "last_activity": latest,
        }
