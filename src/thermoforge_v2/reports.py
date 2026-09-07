"""V2 事实报告：来源 → 想法 → 实验 → 发现 → 决策（V2 实施计划）。

只解释已登记的研究事实，不生成研究结论、不读取隐藏留出、不依赖内部思维链。
JSON、Markdown 和 HTML 共用 sections；WebUI 文档适配器按需导入。
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from html import escape
import json
import math
from typing import Any, Mapping
from urllib.parse import urlsplit

from .report_flow import build_flow
from .report_flow_html import render_flow_html


_DIRECTIONS = {"CVRMSE": "min", "RMSE": "min", "MAE": "min", "MAPE": "min",
               "R2": "max", "R²": "max", "physics_violation_rate": "min"}
_ORIGINS = {"literature": "文献启发", "history": "历史结果", "conjecture": "自主假说",
            "mixed": "多来源组合"}
_FAILURES = {"hypothesis": "假说未获支持", "implementation": "实现失败",
             "data": "数据不足", "budget": "预算不足", "cancelled": "取消",
             "timeout": "超时"}


def _records(value: Any) -> list[dict[str, Any]]:
    return [deepcopy(item) for item in value or [] if isinstance(item, dict)]


def _text(value: Any) -> str:
    if value is None or value == "":
        return "未记录"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _ids(record: Mapping[str, Any], field: str) -> list[str]:
    value = record.get(field) or []
    return [str(item) for item in value] if isinstance(value, (list, tuple)) else []


def _metric_value(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _safe_url(value: Any) -> str:
    url = str(value or "").strip()
    try:
        parsed = urlsplit(url)
        return url if parsed.scheme in {"http", "https"} and parsed.netloc else ""
    except ValueError:
        return ""


def _section(title: str, paragraphs: list[str] | None = None,
             rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"title": title, "paragraphs": paragraphs or [], "rows": rows or []}


def _comparison(jobs: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    """缺少口径、评价面或指标的实验不参与排名；没有从失败中推导零分。"""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded = []
    expected = protocol.get("fingerprint")
    for job in jobs:
        result = job.get("result") or {}
        fingerprint = result.get("protocol_fingerprint")
        reasons = []
        if not expected:
            reasons.append("当前运行缺少评价协议指纹")
        if job.get("status") not in {"completed", "succeeded"}:
            reasons.append("实验尚未成功完成")
        if not fingerprint:
            reasons.append("缺少实验评价协议指纹")
        elif expected and fingerprint != expected:
            reasons.append("实验与当前运行的评价协议不同")
        if result.get("feedback_surface") != "validate":
            reasons.append("未明确记录 validate 评价面")
        metrics = {str(k): v for k, raw in (result.get("metrics") or {}).items()
                   if (v := _metric_value(raw)) is not None}
        if not metrics:
            reasons.append("缺少有效实测指标")
        if reasons:
            excluded.append({"job_id": job.get("id"), "reasons": reasons})
            continue
        groups[str(fingerprint)].append({"job_id": job.get("id"),
                                         "track_id": job.get("track_id"),
                                         "experiment_id": result.get("experiment_id"),
                                         "metrics": metrics})
    compared = []
    for fingerprint, rows in groups.items():
        best = {}
        for name, direction in _DIRECTIONS.items():
            available = [(row["job_id"], row["metrics"][name]) for row in rows
                         if name in row["metrics"]]
            if not available:
                continue
            optimum = (min if direction == "min" else max)(v for _, v in available)
            best[name] = {"direction": direction, "value": optimum,
                          "job_ids": [jid for jid, value in available if value == optimum]}
        compared.append({"protocol_fingerprint": fingerprint, "rows": rows,
                         "best_per_metric": best})
    return {"groups": compared, "excluded": excluded,
            "note": "只比较相同协议指纹下的实测 validate 指标。单指标最优不等于验收通过或路线保留；"
                    "失败、未完成和口径不明的实验不按零分参与排名。留出结果不用于搜索反馈。"}


def _usage(tracks: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    totals: dict[str, int | float | None] = {}
    for track in tracks:
        usage = track.get("usage") or {}
        codex_total = usage.get("total") or {}
        row = {"track_id": track.get("track_id"), "role": track.get("role")}
        for name, alias, backend_name in (("input_tokens", "prompt_tokens", "inputTokens"),
                                         ("output_tokens", "completion_tokens", "outputTokens"),
                                         ("total_tokens", "total_tokens", "totalTokens"),
                                         ("cost", "cost", "cost")):
            raw = usage.get(name, usage.get(alias, codex_total.get(backend_name)))
            row[name] = raw if _metric_value(raw) is not None else None
        if row["total_tokens"] is None and row["input_tokens"] is not None and row["output_tokens"] is not None:
            row["total_tokens"] = row["input_tokens"] + row["output_tokens"]
        rows.append(row)
    for name in ("input_tokens", "output_tokens", "total_tokens", "cost"):
        values = [row[name] for row in rows]
        totals[name] = sum(values) if values and all(v is not None for v in values) else None
    return {"tracks": rows, "totals": totals,
            "note": "主智能体、全部候选的用量均列出；缺失用量或费用表示不可得，不等于零。"}


def build_report(snapshot: Mapping[str, Any], track_id: str | None = None) -> dict[str, Any]:
    """从单次一致性快照生成可追溯报告；不修改快照，也不启动外部请求。"""
    run = deepcopy(dict(snapshot.get("run") or {}))
    protocol = deepcopy(run.get("protocol") or {})
    tracks = _records(snapshot.get("tracks"))
    if track_id is not None:
        tracks = [t for t in tracks if t.get("track_id") == track_id]
        if not tracks:
            raise ValueError(f"没有这条研究轨迹：{track_id}")
    selected = {t.get("track_id") for t in tracks}
    records = {}
    for name in ("ideas", "jobs", "findings", "decisions", "reports"):
        rows = _records(snapshot.get(name))
        records[name] = [r for r in rows if track_id is None or r.get("track_id") in selected]
    sources = _records(snapshot.get("sources"))
    referenced = {sid for idea in records["ideas"] for sid in _ids(idea, "source_ids")}
    if track_id is not None:
        sources = [s for s in sources if s.get("track_id") == track_id or s.get("id") in referenced]
    source_map = {str(s.get("id")): s for s in sources}
    nodes = [{"id": t.get("track_id"), "kind": "track"} for t in tracks]
    edges = []
    missing = []
    for name, rows in {"sources": sources, **records}.items():
        nodes.extend({"id": row.get("id"), "kind": name} for row in rows if row.get("id"))
    known = {str(node["id"]) for node in nodes}
    links = {"source_ids": "source", "parent_idea_ids": "parent_idea",
             "parent_job_ids": "parent_experiment", "idea_ids": "idea",
             "job_ids": "experiment", "finding_ids": "finding", "evidence_ids": "evidence"}
    for name, rows in records.items():
        for row in rows:
            refs = {field: _ids(row, field) for field in links}
            if row.get("idea_id"):
                refs["idea_ids"].append(str(row["idea_id"]))
            for field, ids in refs.items():
                for ref in ids:
                    edges.append({"from": ref, "to": row.get("id"), "relation": links[field]})
                    if ref not in known:
                        missing.append({"record_id": row.get("id"), "reference_id": ref})

    comparison = _comparison(records["jobs"], protocol)
    final_evaluation = deepcopy(snapshot.get("final_evaluation"))
    if final_evaluation and track_id is not None and final_evaluation.get("track_id_selected") not in {None, track_id}:
        final_evaluation = {"status": "not_selected", "reason": "本轨迹不是该运行已选定的最终留出候选；本报告不将其他路线的留出结果归给本轨迹。"}
    negatives = []
    for job in records["jobs"]:
        result = job.get("result") or {}
        if job.get("status") in {"failed", "error", "cancelled", "interrupted", "timeout"}:
            category = result.get("failure_category") or job.get("failure_category") or job.get("status")
            negatives.append({"job_id": job.get("id"), "track_id": job.get("track_id"),
                              "kind": category, "reason": result.get("error") or job.get("error"),
                              "interpretation": "执行未产生有效验证结果；不能据此认定假说被证伪。"})
    for finding in records["findings"]:
        if finding.get("failure_category") not in {None, "", "none"}:
            negatives.append({"finding_id": finding.get("id"), "track_id": finding.get("track_id"),
                              "kind": finding["failure_category"], "reason": finding.get("statement"),
                              "interpretation": finding.get("interpretation")})

    sections = [_section("运行概览", [
        f"研究运行：{_text(run.get('run_id'))}；状态：{_text(run.get('status'))}；"
        f"目标：{_text((run.get('config') or {}).get('goal_id') or protocol.get('goal_id'))}。",
        f"本报告包含 {len(tracks)} 条轨迹、{len(records['ideas'])} 个想法、"
        f"{len(records['jobs'])} 个实验请求、{len(sources)} 项来源。",
        "研究理由来自实验前登记的想法记录；观察来自实验结果；解释与路线决策保留各自作者和依据。",
    ])]
    sections.append(_section("评价口径与可比较性", [comparison["note"],
        f"协议指纹：{_text(protocol.get('fingerprint'))}",
        f"数据修订：{_text(protocol.get('dataset_ref'))}；"
        f"验证方案：{_text(protocol.get('validation'))}；"
        f"指标配置：{_text(protocol.get('metrics'))}；"
        f"物理约束：{_text(protocol.get('physics_tests'))}。",
    ], [{"实验请求": item["job_id"], "不参与排名原因": "；".join(item["reasons"])}
        for item in comparison["excluded"]]))
    for group in comparison["groups"]:
        sections.append(_section("同口径实测结果", [
            f"协议：{group['protocol_fingerprint']}；评价面：validate。",
            "逐指标最好值：" + _text(group["best_per_metric"]),
        ], [{"轨迹": row["track_id"], "请求": row["job_id"],
             "实验": row["experiment_id"], "指标": _text(row["metrics"])} for row in group["rows"]]))

    if final_evaluation:
        if final_evaluation.get("status") == "evaluated":
            surfaces = final_evaluation.get("surfaces") or {}
            sections.append(_section("最终留出评价（仅供外部查看）", [
                "该评价由外部报告服务提供，与搜索使用的 validate 指标分别展示；不反馈给内部研究智能体。",
                f"选定请求：{_text(final_evaluation.get('job_id'))}；实验：{_text(final_evaluation.get('experiment_id'))}；"
                f"轨迹：{_text(final_evaluation.get('track_id_selected'))}。",
                "选择依据：" + _text(final_evaluation.get("selection_reason")),
                "冻结实验报告哈希：" + _text(final_evaluation.get("report_sha256")),
                "物理评价：" + _text(final_evaluation.get("physics")),
                "已有留出面的实测结果如下。" if surfaces else "未提供可用留出面结果，不能判断最终留出是否达标。",
                "这些指标本身不表示模型已经批准或发布。",
            ], [{"留出面": surface, "结果": _text(value)} for surface, value in surfaces.items()]))
        else:
            sections.append(_section("最终留出评价", [_text(final_evaluation.get("reason"))]))

    for track in tracks:
        tid = track.get("track_id")
        track_ideas = [idea for idea in records["ideas"] if idea.get("track_id") == tid]
        paragraphs = [f"身份：{_text(track.get('role'))}；状态：{_text(track.get('status'))}；"
                      f"阶段：{_text(track.get('phase'))}；完成执行片段：{_text(track.get('turns'))}。"]
        if track.get("error"):
            paragraphs.append("运行错误：" + _text(track["error"]))
        if not track_ideas:
            paragraphs.append("尚无已登记想法，不能据此报告已完成研究或已有结论。")
        sections.append(_section(f"研究轨迹 {tid}", paragraphs))
        for idea in track_ideas:
            iid = idea.get("id")
            paragraphs = [f"想法 {iid}：{_text(idea.get('statement'))}",
                          "来源类型：" + _ORIGINS.get(idea.get("origin"), _text(idea.get("origin"))),
                          "选择这个方向的理由：" + _text(idea.get("reason")),
                          "实验前预测：" + _text(idea.get("prediction")),
                          "反证条件：" + _text(idea.get("falsification"))]
            for sid in _ids(idea, "source_ids"):
                source = source_map.get(sid)
                paragraphs.append(f"依据 {sid}：" + (f"{_text(source.get('title'))}；"
                    f"实际阅读范围：{_text(source.get('read_scope'))}。" if source else "引用的来源记录缺失。"))
            if idea.get("origin") == "conjecture" and not _ids(idea, "source_ids"):
                paragraphs.append("该方向登记为自主假说，未声称由论文支持。")
            if _ids(idea, "parent_idea_ids") or _ids(idea, "parent_job_ids"):
                paragraphs.append("历史父节点：" + "、".join(
                    _ids(idea, "parent_idea_ids") + _ids(idea, "parent_job_ids")))
            related = [j for j in records["jobs"] if j.get("idea_id") == iid]
            rows = []
            for job in related:
                result = job.get("result") or {}
                rows.append({"请求": job.get("id"), "实验": result.get("experiment_id"),
                             "状态": job.get("status"), "实际模型": _text(result.get("model") or
                                 (job.get("request") or {}).get("model")),
                             "实测指标": _text(result.get("metrics")),
                             "错误": _text(result.get("error") or job.get("error"))})
            if not related:
                paragraphs.append("尚无关联实验，预测未经实验验证。")
            sections.append(_section(f"想法与实验 {iid}", paragraphs, rows))
        for finding in records["findings"]:
            if finding.get("track_id") != tid:
                continue
            sections.append(_section(f"发现 {finding.get('id')}", [
                "观察：" + _text(finding.get("statement")),
                "研究解释：" + _text(finding.get("interpretation")),
                "依据实验：" + "、".join(_ids(finding, "job_ids")),
                "适用条件与限制：" + _text(finding.get("limitations")),
            ]))

    sections.append(_section("路线保留与淘汰", [
        "以下是已登记决定；未登记决定的路线不由报告器自动淘汰。"
        if records["decisions"] else "尚无已登记的保留或淘汰决定。"], [
        {"决定": d.get("id"), "轨迹": d.get("track_id"),
         "动作": d.get("action") or d.get("decision") or d.get("kind"),
         "理由": _text(d.get("reason")), "依据": _text({k: d[k] for k in links if d.get(k)})}
        for d in records["decisions"]]))
    sections.append(_section("负结果与未完成工作", [
        "执行失败与假说被证伪分别记录；预算不足、超时和数据不足均不能代替科学结论。"
        if negatives else "暂无已登记负结果；不代表所有假说均获支持。"], [
        {"记录": n.get("job_id") or n.get("finding_id"), "轨迹": n.get("track_id"),
         "类别": _FAILURES.get(n["kind"], n["kind"]), "原因": _text(n["reason"]),
         "解释": _text(n["interpretation"])} for n in negatives]))
    for narrative in records["reports"]:
        sections.append(_section(f"智能体报告：{narrative.get('title') or narrative.get('id')}", [
            f"记录：{narrative.get('id')}；作者轨迹：{narrative.get('track_id')}。",
            "摘要：" + _text(narrative.get("summary")), _text(narrative.get("body")),
            "依据：" + _text({k: narrative[k] for k in links if narrative.get(k)}),
            "限制：" + _text(narrative.get("limitations")),
        ]))
    usage = _usage(tracks)
    sections.append(_section("运行用量", [usage["note"]], usage["tracks"]))
    sections.append(_section("来源登记", ["read_scope 表示实际读取范围；metadata 不能当作阅读全文。"], [
        {"来源": s.get("id"), "标题": s.get("title"), "类型": s.get("kind"),
         "阅读范围": s.get("read_scope"), "URL": _safe_url(s.get("url")),
         "DOI": s.get("doi"), "登记核验": s.get("verification"),
         "检索时间": s.get("retrieved_at"), "内容哈希": s.get("content_sha256")}
        for s in sources]))
    if missing:
        sections.append(_section("引用完整性", ["以下引用未在当前报告快照中找到，依据尚不完整。"], missing))

    report = {"schema_version": "thermoforge.research-report.v2", "run_id": run.get("run_id"),
              "track_id": track_id, "status": run.get("status"),
              "title": f"ThermoForge V2 {'轨迹' if track_id else '综合'}研究报告",
              "generated_at": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
              "summary": {"tracks": len(tracks), "ideas": len(records["ideas"]),
                          "jobs": len(records["jobs"]), "sources": len(sources)},
              "sections": sections, "sources": sources, **records,
              "tracks": tracks, "comparability": comparison, "usage": usage,
              "negative_results": negatives,
              "final_evaluation": final_evaluation,
              "flow": build_flow(snapshot, track_id=track_id),
              "lineage": {"nodes": nodes, "edges": edges, "missing_references": missing}}
    return report


def _md_cell(value: Any) -> str:
    return _text(value).replace("|", "\\|").replace("\r", "").replace("\n", "<br>")


def render_markdown(report: Mapping[str, Any]) -> str:
    """导出有稳定记录 ID 的 Markdown；原始研究叙述按文本保留。"""
    lines = [f"# {_text(report.get('title'))}", "",
             f"运行：{_text(report.get('run_id'))} · 生成时间：{_text(report.get('generated_at'))}", ""]
    for section in report.get("sections") or []:
        lines.extend([f"## {section['title']}", ""])
        for paragraph in section.get("paragraphs") or []:
            lines.extend([str(paragraph), ""])
        rows = section.get("rows") or []
        if rows:
            columns = list(dict.fromkeys(key for row in rows for key in row))
            lines.extend(["| " + " | ".join(_md_cell(k) for k in columns) + " |",
                          "| " + " | ".join("---" for _ in columns) + " |"])
            lines.extend("| " + " | ".join(_md_cell(row.get(k)) for k in columns) + " |" for row in rows)
            lines.append("")
    return "\n".join(lines)


def render_html(report: Mapping[str, Any]) -> str:
    """轻量自包含 HTML。所有研究文本转义，不执行模型/资料中的 HTML。"""
    body = [f"<h1>{escape(_text(report.get('title')))}</h1>",
            f"<p>运行：{escape(_text(report.get('run_id')))} · {escape(_text(report.get('generated_at')))}</p>"]
    flow_html = render_flow_html(report.get("flow"))
    if flow_html:
        body.extend([flow_html, "<details class='full-report'><summary>完整文字报告与证据记录</summary>"])
    for section in report.get("sections") or []:
        body.append(f"<section><h2>{escape(str(section['title']))}</h2>")
        body.extend(f"<p>{escape(str(p))}</p>" for p in section.get("paragraphs") or [])
        rows = section.get("rows") or []
        if rows:
            columns = list(dict.fromkeys(key for row in rows for key in row))
            body.append("<div class='table'><table><thead><tr>" + "".join(
                f"<th>{escape(str(k))}</th>" for k in columns) + "</tr></thead><tbody>")
            for row in rows:
                cells = []
                for key in columns:
                    value = row.get(key)
                    url = _safe_url(value) if key == "URL" else ""
                    text = (f'<a href="{escape(url, quote=True)}" rel="noopener noreferrer">'
                            f'{escape(url)}</a>' if url else escape(_text(value)))
                    cells.append(f"<td>{text}</td>")
                body.append("<tr>" + "".join(cells) + "</tr>")
            body.append("</tbody></table></div>")
        body.append("</section>")
    if flow_html:
        body.append("</details>")
    return ("<!doctype html><html lang='zh-CN'><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{escape(_text(report.get('title')))}</title><style>"
            "body{max-width:1540px;margin:32px auto;padding:0 24px;font:16px/1.7 system-ui;"
            "color:#202832;background:#fff}h1{font-size:28px}h2{font-size:21px;margin-top:36px;"
            "border-top:1px solid #dce2e8;padding-top:20px}p{white-space:pre-wrap;overflow-wrap:anywhere}"
            ".table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}"
            "th,td{border:1px solid #dce2e8;padding:8px;text-align:left;vertical-align:top;"
            "overflow-wrap:anywhere}th{background:#f4f6f8}"
            ".full-report>summary{cursor:pointer;font-size:17px;padding:12px 0;color:#285e4c}"
            "@media(max-width:600px){body{padding:0 12px;margin:20px auto}h1{font-size:23px}}"
            "</style><body>" + "".join(body) + "</body></html>")


def to_document(report: Mapping[str, Any]):
    """可选适配既有 HTML/Markdown/PDF 导出器，后台服务不必导入 UI 依赖。"""
    import pandas as pd
    from thermoforge_webui.services.reports import ReportDocument, Section

    return ReportDocument(title=str(report.get("title") or "V2 研究报告"),
                          subtitle=f"运行 {report.get('run_id')}",
                          generated_at=str(report.get("generated_at") or ""),
                          sections=[Section(title=s["title"], paragraphs=list(s.get("paragraphs") or []),
                                            table=pd.DataFrame(s["rows"]) if s.get("rows") else None)
                                    for s in report.get("sections") or []],
                          footer="由 ThermoForge V2 已登记事实生成；缺失信息不作为零或成功。")
