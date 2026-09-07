"""持久控制平面：SQLite 事务、幂等消息、预算预留和研究证据。

每次事务独占连接，兼容后台线程和多个客户端；V1 账本与数据 Vault 不变。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .contracts import MUTABLE_CONFIG, RunConfig, V2Error
from .autonomy import AutonomyMixin


def now() -> float:
    return time.time()


def encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":"))


def identifier(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


class RunStore(AutonomyMixin):
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "state.sqlite3"
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, idempotency_key TEXT UNIQUE NOT NULL,
                    request_hash TEXT NOT NULL, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS tracks (
                    run_id TEXT NOT NULL, id TEXT NOT NULL, data TEXT NOT NULL,
                    PRIMARY KEY(run_id,id));
                CREATE TABLE IF NOT EXISTS records (
                    id TEXT PRIMARY KEY, run_id TEXT NOT NULL, kind TEXT NOT NULL,
                    track_id TEXT, data TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS record_lookup ON records(run_id,kind,track_id);
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
                    track_id TEXT, kind TEXT NOT NULL, payload TEXT NOT NULL, at REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS event_lookup ON events(run_id,seq);
                CREATE TABLE IF NOT EXISTS job_keys (
                    run_id TEXT, track_id TEXT, key TEXT, job_id TEXT,
                    request_hash TEXT, PRIMARY KEY(run_id,track_id,key));
                CREATE TABLE IF NOT EXISTS leases (
                    scope TEXT PRIMARY KEY, owner TEXT NOT NULL, expires REAL NOT NULL);
            """)

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA busy_timeout=30000")
        db.execute("PRAGMA journal_mode=WAL")
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _get(db, table, key, value):
        row = db.execute(f"SELECT data FROM {table} WHERE {key}=?", (value,)).fetchone()
        if row is None:
            raise V2Error("TFV2-NOT-FOUND", f"找不到 {value}")
        return json.loads(row[0])

    def create_run(self, config: dict, protocol: dict, idempotency_key: str) -> dict:
        if not idempotency_key or len(idempotency_key) > 200:
            raise V2Error("TFV2-INPUT", "必须提供长度 1–200 的幂等键")
        existing_request = self.find_request(config, idempotency_key)
        if existing_request is not None:
            return existing_request
        config = RunConfig.model_validate(config).model_dump(mode="json")
        request_hash = hashlib.sha256(encode(config).encode()).hexdigest()
        with self._db() as db:
            existing = db.execute("SELECT request_hash,data FROM runs WHERE idempotency_key=?",
                                  (idempotency_key,)).fetchone()
            if existing:
                if existing[0] != request_hash:
                    raise V2Error("TFV2-CONFLICT", "同一幂等键已用于不同的研究配置")
                return json.loads(existing[1])
            run_id = identifier("RUN")
            doc = {"run_id": run_id, "config": config, "protocol": protocol,
                   "status": "queued", "version": 1, "created_at": now(),
                   "updated_at": now(), "error": None, "experiments_reserved": 0,
                   "experiments_settled": 0, "tokens_used": 0}
            if config["research_mode"] == "autonomous":
                doc["research_stage"] = "independent_proposals"
            db.execute("INSERT INTO runs VALUES(?,?,?,?)",
                       (run_id, idempotency_key, request_hash, encode(doc)))
            self._event(db, run_id, "created", {"config": config})
        return doc

    def get_run(self, run_id: str) -> dict:
        with self._db() as db:
            return self._get(db, "runs", "id", run_id)

    def find_request(self, config: dict, idempotency_key: str) -> dict | None:
        """重试直接返回原运行，不重新解释可能已经改变的环境或研究目标。"""
        if not idempotency_key or len(idempotency_key) > 200:
            raise V2Error("TFV2-INPUT", "必须提供长度 1–200 的幂等键")
        with self._db() as db:
            row = db.execute("SELECT request_hash,data FROM runs WHERE idempotency_key=?",
                             (idempotency_key,)).fetchone()
            if row is None:
                return None
            # 原始请求哈希不随 guidance/预算更新改变。旧记录新增字段之前的
            # 哈希也用原请求重建，不能拿已被用户更新的运行配置来判断幂等。
            if config.get("research_mode", "acceptance") == "acceptance" and not config.get("reuse_experiments", False):
                legacy = RunConfig.model_validate(config | {"research_mode": "acceptance"}).model_dump(mode="json")
                legacy.pop("research_mode")
                legacy.pop("reuse_experiments")
                if hashlib.sha256(encode(legacy).encode()).hexdigest() == row[0]:
                    return json.loads(row[1])
            config = RunConfig.model_validate(config).model_dump(mode="json")
            request_hash = hashlib.sha256(encode(config).encode()).hexdigest()
            if row[0] != request_hash:
                raise V2Error("TFV2-CONFLICT", "同一幂等键已用于不同的研究配置")
            return json.loads(row[1])

    def begin_run(self, run_id: str) -> dict:
        """调度领取和用户暂停共用事务，已暂停的队列任务不得重新启动。"""
        with self._db() as db:
            doc = self._get(db, "runs", "id", run_id)
            if doc["status"] != "queued":
                return doc | {"started": False}
            doc.update(status="running", error=None)
            self._write_run(db, doc)
            self._event(db, run_id, "run.started", {})
            return doc | {"started": True}

    def list_runs(self) -> list[dict]:
        with self._db() as db:
            docs = [json.loads(row[0]) for row in db.execute("SELECT data FROM runs")]
        return sorted(docs, key=lambda x: x["created_at"], reverse=True)

    @staticmethod
    def _write_run(db, doc):
        doc["version"] += 1
        doc["updated_at"] = now()
        db.execute("UPDATE runs SET data=? WHERE id=?", (encode(doc), doc["run_id"]))

    def update_run(self, run_id: str, patch: dict, expected_version: int | None = None) -> dict:
        with self._db() as db:
            doc = self._get(db, "runs", "id", run_id)
            if expected_version is not None and doc["version"] != expected_version:
                raise V2Error("TFV2-CONFLICT", "研究状态已改变，请读取新版本后重试")
            if set(patch) & {"run_id", "protocol", "created_at", "version"}:
                raise V2Error("TFV2-IMMUTABLE", "研究身份与评价协议不可修改")
            doc.update(patch)
            self._write_run(db, doc)
            return doc

    def control(self, run_id: str, action: str, expected_version=None, changes=None) -> dict:
        with self._db() as db:
            doc = self._get(db, "runs", "id", run_id)
            if expected_version is not None and doc["version"] != expected_version:
                raise V2Error("TFV2-CONFLICT", "研究状态已改变，请刷新后重试")
            state = doc["status"]
            if action == "pause":
                if state in {"running", "pausing"}:
                    doc["status"] = "pausing"
                elif state == "queued":
                    doc["status"] = "paused"
                elif state != "paused":
                    raise V2Error("TFV2-STATE", f"{state} 不可暂停")
            elif action == "resume":
                if state not in {"paused", "interrupted", "needs_input", "budget_exhausted"}:
                    raise V2Error("TFV2-STATE", f"{state} 不可恢复")
                doc["status"], doc["error"] = "queued", None
            elif action == "cancel":
                if state == "completed":
                    raise V2Error("TFV2-STATE", "已完成研究不能取消")
                doc["status"] = "cancelling" if state in {"running", "pausing", "cancelling"} else "cancelled"
            elif action == "update":
                changes = changes or {}
                if not changes or set(changes) - MUTABLE_CONFIG:
                    raise V2Error("TFV2-INPUT", "仅可调整指导与预算；评价协议和候选拓扑固定")
                if state in {"completed", "cancelled"}:
                    raise V2Error("TFV2-STATE", "已结束研究不能修改")
                config = RunConfig.model_validate({"research_mode": "acceptance", **doc["config"], **changes}).model_dump(mode="json")
                if config["max_experiments"] < doc["experiments_reserved"]:
                    raise V2Error("TFV2-BUDGET", "实验上限不能低于已预留数量")
                doc["config"] = config
            else:
                raise V2Error("TFV2-INPUT", f"未知控制动作 {action}")
            self._write_run(db, doc)
            self._event(db, run_id, "control", {"action": action, "changes": changes,
                                                "status": doc["status"]})
            return doc

    def create_track(self, run_id: str, track_id: str, role: str, **fields) -> dict:
        with self._db() as db:
            self._get(db, "runs", "id", run_id)
            row = db.execute("SELECT data FROM tracks WHERE run_id=? AND id=?",
                             (run_id, track_id)).fetchone()
            if row:
                return json.loads(row[0])
            doc = {"run_id": run_id, "track_id": track_id, "role": role,
                   "instance_id": identifier("AGENT"), "status": "pending",
                   "phase": "prepare", "turns": 0, "failures": 0, "usage": {},
                   "pid": None, "thread_id": None, "session_id": None, "error": None,
                   "created_at": now(), "updated_at": now(), **fields}
            db.execute("INSERT INTO tracks VALUES(?,?,?)", (run_id, track_id, encode(doc)))
            return doc

    def get_track(self, run_id: str, track_id: str) -> dict:
        with self._db() as db:
            row = db.execute("SELECT data FROM tracks WHERE run_id=? AND id=?",
                             (run_id, track_id)).fetchone()
            if not row:
                raise V2Error("TFV2-NOT-FOUND", f"找不到研究轨迹 {track_id}")
            return json.loads(row[0])

    def list_tracks(self, run_id: str) -> list[dict]:
        with self._db() as db:
            return [json.loads(row[0]) for row in db.execute(
                "SELECT data FROM tracks WHERE run_id=? ORDER BY id", (run_id,))]

    def update_track(self, run_id: str, track_id: str, patch: dict) -> dict:
        with self._db() as db:
            row = db.execute("SELECT data FROM tracks WHERE run_id=? AND id=?",
                             (run_id, track_id)).fetchone()
            if row is None:
                raise V2Error("TFV2-NOT-FOUND", track_id)
            doc = json.loads(row[0])
            if set(patch) & {"run_id", "track_id", "created_at"}:
                raise V2Error("TFV2-IMMUTABLE", "轨迹身份不可修改")
            doc.update(patch)
            doc["updated_at"] = now()
            db.execute("UPDATE tracks SET data=? WHERE run_id=? AND id=?",
                       (encode(doc), run_id, track_id))
            return doc

    def add_record(self, run_id: str, kind: str, record: dict, track_id=None, record_id=None) -> dict:
        with self._db() as db:
            self._get(db, "runs", "id", run_id)
            doc = dict(record)
            doc.update(id=record_id or identifier(kind.upper().rstrip("S")),
                       run_id=run_id, track_id=track_id, created_at=now())
            try:
                db.execute("INSERT INTO records VALUES(?,?,?,?,?)",
                           (doc["id"], run_id, kind, track_id, encode(doc)))
            except sqlite3.IntegrityError as exc:
                raise V2Error("TFV2-IMMUTABLE", "证据 ID 已存在，不能覆盖") from exc
            self._event(db, run_id, f"{kind}.created", {"id": doc["id"]}, track_id)
            return doc

    def records(self, run_id: str, kind: str, track_id=None) -> list[dict]:
        query, args = "SELECT data FROM records WHERE run_id=? AND kind=?", [run_id, kind]
        if track_id is not None:
            query += " AND track_id=?"
            args.append(track_id)
        with self._db() as db:
            return [json.loads(row[0]) for row in db.execute(query + " ORDER BY rowid", args)]

    def update_record(self, run_id: str, kind: str, record_id: str, patch: dict) -> dict:
        if kind not in {"jobs", "messages", "reports"}:
            raise V2Error("TFV2-IMMUTABLE", "来源、想法和发现需追加新证据，不能改写历史")
        with self._db() as db:
            doc = self._get(db, "records", "id", record_id)
            row = db.execute("SELECT kind FROM records WHERE id=?", (record_id,)).fetchone()
            if doc["run_id"] != run_id or row[0] != kind:
                raise V2Error("TFV2-NOT-FOUND", record_id)
            if set(patch) & {"id", "run_id", "track_id", "created_at"}:
                raise V2Error("TFV2-IMMUTABLE", "记录身份不可修改")
            doc.update(patch, updated_at=now())
            db.execute("UPDATE records SET data=? WHERE id=?", (encode(doc), record_id))
            return doc

    def reserve_job(self, run_id, track_id, idea_id, request, idempotency_key) -> dict:
        if not idempotency_key or len(idempotency_key) > 200:
            raise V2Error("TFV2-INPUT", "实验必须提供幂等键")
        digest = hashlib.sha256(encode({"idea_id": idea_id, "request": request}).encode()).hexdigest()
        with self._db() as db:
            found = db.execute("SELECT job_id,request_hash FROM job_keys WHERE run_id=? AND track_id=? AND key=?",
                               (run_id, track_id, idempotency_key)).fetchone()
            if found:
                if found[1] != digest:
                    raise V2Error("TFV2-CONFLICT", "实验幂等键已用于另一请求")
                return self._get(db, "records", "id", found[0]) | {"fresh": False}
            run = self._get(db, "runs", "id", run_id)
            if run["status"] != "running":
                raise V2Error("TFV2-STATE", "研究当前不接受新实验")
            if not db.execute("SELECT 1 FROM tracks WHERE run_id=? AND id=?", (run_id, track_id)).fetchone():
                raise V2Error("TFV2-NOT-FOUND", track_id)
            proposal = None
            if run["config"].get("research_mode", "acceptance") == "autonomous":
                if not self._advance_autonomy(db, run)["proposal_barrier_open"]:
                    raise V2Error("TFV2-PHASE", "首轮提案尚未齐备；结束当前回合，由软件等待并继续")
                if db.execute("SELECT 1 FROM records WHERE run_id=? AND kind='stops' AND track_id=?",
                              (run_id, track_id)).fetchone():
                    raise V2Error("TFV2-STATE", "轨迹已有停止决定，不能继续实验")
                row = db.execute("SELECT data FROM records WHERE id=? AND run_id=? AND track_id=? AND kind='proposals'",
                                 (request.get("proposal_id"), run_id, track_id)).fetchone()
                proposal = json.loads(row[0]) if row else None
                if (not proposal or proposal["idea_id"] != idea_id or proposal["model"] != request.get("model")
                        or proposal["experiment_fingerprint"] != request.get("experiment_fingerprint")
                        or proposal.get("lab_content_hash") != request.get("lab_content_hash")
                        or proposal["protocol_fingerprint"] != run["protocol"]["fingerprint"]
                        or request.get("protocol_fingerprint") != run["protocol"]["fingerprint"]):
                    raise V2Error("TFV2-PROPOSAL", "实验必须匹配本轨迹的冻结提案、模型与评价协议")
                previous = next((json.loads(r[0]) for r in db.execute(
                    "SELECT data FROM records WHERE run_id=? AND track_id=? AND kind='jobs'", (run_id, track_id))
                    if json.loads(r[0]).get("proposal_id") == proposal["id"]), None)
                if previous:
                    db.execute("INSERT INTO job_keys VALUES(?,?,?,?,?)", (run_id, track_id, idempotency_key, previous["id"], digest))
                    return previous | {"fresh": False}
                track = json.loads(db.execute("SELECT data FROM tracks WHERE run_id=? AND id=?", (run_id, track_id)).fetchone()[0])
                if track.get("turns", 0) > 0 and any(json.loads(r[0]).get("research_turn") == track["turns"] for r in db.execute(
                        "SELECT data FROM records WHERE run_id=? AND track_id=? AND kind='jobs'", (run_id, track_id))):
                    raise V2Error("TFV2-PHASE", "本回合已执行实验；保存发现并结束回合，软件随后推进下一实验")
            count = db.execute("SELECT COUNT(*) FROM records WHERE run_id=? AND kind='jobs' AND track_id=?",
                               (run_id, track_id)).fetchone()[0]
            if (run["experiments_reserved"] >= run["config"]["max_experiments"]
                    or count >= run["config"]["max_experiments_per_track"]):
                raise V2Error("TFV2-BUDGET", "实验预算已用完（失败与已预留作业也计入）")
            doc = {"id": identifier("JOB"), "run_id": run_id, "track_id": track_id,
                   "idea_id": idea_id, "request": request, "status": "reserved",
                   "created_at": now(), "result": None,
                   "protocol_fingerprint": run["protocol"]["fingerprint"]}
            if proposal:
                doc.update(proposal_id=proposal["id"], experiment_fingerprint=proposal["experiment_fingerprint"],
                           purpose=proposal["purpose"], executed=False, research_turn=track.get("turns", 0))
            db.execute("INSERT INTO records VALUES(?,?,?,?,?)", (doc["id"], run_id, "jobs", track_id, encode(doc)))
            db.execute("INSERT INTO job_keys VALUES(?,?,?,?,?)", (run_id, track_id, idempotency_key, doc["id"], digest))
            run["experiments_reserved"] += 1
            self._write_run(db, run)
            self._event(db, run_id, "job.reserved", {"id": doc["id"]}, track_id)
            return doc | {"fresh": True}

    def settle_job(self, job_id: str, status: str, result: dict) -> dict:
        if status not in {"completed", "failed", "cancelled", "interrupted"}:
            raise V2Error("TFV2-INPUT", "无效的实验终态")
        with self._db() as db:
            doc = self._get(db, "records", "id", job_id)
            if doc["status"] in {"completed", "failed", "cancelled", "interrupted"}:
                return doc
            doc.update(status=status, result=result, finished_at=now())
            db.execute("UPDATE records SET data=? WHERE id=?", (encode(doc), job_id))
            run = self._get(db, "runs", "id", doc["run_id"])
            run["experiments_settled"] += 1
            self._write_run(db, run)
            self._event(db, doc["run_id"], "job.settled", {"id": job_id, "status": status}, doc["track_id"])
            return doc

    def start_job(self, job_id: str) -> dict:
        """拿到实验槽后原子检查控制状态，消除暂停与开始训练之间的竞态。"""
        with self._db() as db:
            doc = self._get(db, "records", "id", job_id)
            if doc["status"] != "reserved":
                return doc | {"started": False}
            run = self._get(db, "runs", "id", doc["run_id"])
            if run["status"] != "running":
                doc.update(status="cancelled", finished_at=now(), result={
                    "failure_category": "control", "error": "研究已暂停或停止，未开始训练",
                    "training_started": False, "reservation_consumed": True})
                run["experiments_settled"] += 1
                self._write_run(db, run)
                started = False
            else:
                state = self._advance_autonomy(db, run)
                own_feedback = any(json.loads(r[0]).get("status") in {"completed", "failed", "cancelled", "interrupted"}
                    and json.loads(r[0]).get("proposal_id") for r in db.execute(
                        "SELECT data FROM records WHERE run_id=? AND track_id=? AND kind='jobs'",
                        (doc["run_id"], doc["track_id"])))
                if (state["sharing_ready"] and run["config"].get("reuse_experiments")
                        and own_feedback and doc.get("purpose") != "replicate" and doc.get("experiment_fingerprint")):
                    other = next((json.loads(r[0]) for r in db.execute(
                        "SELECT data FROM records WHERE run_id=? AND kind='jobs' ORDER BY rowid", (doc["run_id"],))
                        if json.loads(r[0]).get("status") == "completed"
                        and json.loads(r[0]).get("executed") is True
                        and json.loads(r[0]).get("experiment_fingerprint") == doc["experiment_fingerprint"]), None)
                    if other:
                        doc.update(status="completed", finished_at=now(), executed=False,
                                   reused_from_job_id=other["id"], result=other["result"],
                                   experiment_id=other.get("experiment_id") or (other.get("result") or {}).get("experiment_id"))
                        run["experiments_settled"] += 1
                        self._write_run(db, run)
                        db.execute("UPDATE records SET data=? WHERE id=?", (encode(doc), doc["id"]))
                        self._event(db, doc["run_id"], "job.reused", {"id": doc["id"], "reused_from_job_id": other["id"]}, doc["track_id"])
                        return doc | {"started": False, "reused": True}
                doc.update(status="running", started_at=now())
                if doc.get("proposal_id"):
                    doc["executed"] = True
                started = True
            db.execute("UPDATE records SET data=? WHERE id=?", (encode(doc), job_id))
            self._event(db, doc["run_id"], "job.started" if started else "job.cancelled",
                        {"id": job_id}, doc["track_id"])
            return doc | {"started": started}

    @staticmethod
    def _event(db, run_id, kind, payload, track_id=None):
        return db.execute("INSERT INTO events(run_id,track_id,kind,payload,at) VALUES(?,?,?,?,?)",
                          (run_id, track_id, kind, encode(payload), now())).lastrowid

    def event(self, run_id, kind, payload, track_id=None):
        with self._db() as db:
            return self._event(db, run_id, kind, payload, track_id)

    def events(self, run_id, after=0, limit=100):
        with self._db() as db:
            rows = db.execute("SELECT seq,track_id,kind,payload,at FROM events WHERE run_id=? AND seq>? ORDER BY seq LIMIT ?",
                              (run_id, max(0, after), min(1000, max(1, limit)))).fetchall()
        docs = [{"seq": r[0], "run_id": run_id, "track_id": r[1], "kind": r[2],
                 "payload": json.loads(r[3]), "at": r[4]} for r in rows]
        return {"events": docs, "cursor": docs[-1]["seq"] if docs else after}

    def lease(self, scope: str, owner: str, ttl=30) -> bool:
        with self._db() as db:
            current = db.execute("SELECT owner,expires FROM leases WHERE scope=?", (scope,)).fetchone()
            if current and current[0] != owner and current[1] > now():
                return False
            db.execute("INSERT OR REPLACE INTO leases VALUES(?,?,?)", (scope, owner, now() + ttl))
            return True

    def release_lease(self, scope, owner):
        with self._db() as db:
            db.execute("DELETE FROM leases WHERE scope=? AND owner=?", (scope, owner))

    def clear_leases(self):
        """仅由已经取得服务 OS 独占锁的恢复入口调用。"""
        with self._db() as db:
            db.execute("DELETE FROM leases")

    def snapshot(self, run_id):
        with self._db() as db:
            result = {"run": self._get(db, "runs", "id", run_id), "tracks": [json.loads(row[0])
                for row in db.execute("SELECT data FROM tracks WHERE run_id=? ORDER BY id", (run_id,))]}
            for kind in ("sources", "ideas", "jobs", "findings", "decisions", "messages", "reports", "proposals", "stops"):
                result[kind] = [json.loads(row[0]) for row in db.execute(
                    "SELECT data FROM records WHERE run_id=? AND kind=? ORDER BY rowid", (run_id, kind))]
            # External snapshots may audit repeated work; internal history/team tools
            # read records directly and do not reveal peers to independent candidates.
            first = {}
            for job in result["jobs"]:
                identity = job.get("experiment_fingerprint")
                if identity and job.get("status") == "completed":
                    if identity in first:
                        job["duplicate_of_job_id"] = first[identity]
                    else:
                        first[identity] = job["id"]
            rows = db.execute("SELECT seq,track_id,kind,payload,at FROM events WHERE run_id=? ORDER BY seq DESC LIMIT 1000",
                              (run_id,)).fetchall()
            result["events"] = [{"seq": row[0], "run_id": run_id, "track_id": row[1], "kind": row[2],
                                 "payload": json.loads(row[3]), "at": row[4]} for row in reversed(rows)]
        return result
