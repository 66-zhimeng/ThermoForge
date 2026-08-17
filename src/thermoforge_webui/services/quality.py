"""数据质量：诊断 → 人话解释 → 可执行处方。

这一层要解决的是「数据质量心里没底、不知道怎么处理」。诊断本身由
`modelability.py`（语义门禁）和 `tf_dataset_profile`（统计画像）给出，
这里做的是**把结论翻译成动作**：

- 能由预处理规则库解决的（规则库当前只有三条，都是 legacy 工作簿口径），
  给出可一键生成的 `status=proposed` 规则草案，由人以 `actor=human` 审批。
- **解决不了的必须明说**。派生量循环、缺流量测点这类问题不是「清洗一下
  就好」，硬要预处理反而会造出一个看起来能跑、实则循环论证的模型
  （data-survey §F1 的 load = current_percent × 9672/100 就是这个坑）。
  这里宁可写「预处理解决不了，得改目标口径或补测点」。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from thermoforge_data.preprocess import RULE_LIBRARY
from ..envelope import entries

Severity = Literal["blocker", "warning", "info"]

SEVERITY_ICONS = {"blocker": "🔴", "warning": "🟡", "info": "🟢"}
SEVERITY_LABELS = {"blocker": "阻断", "warning": "警告", "info": "通过"}

# 缺失率超过这个比例就提出来说——不是硬门禁，是让人知道有这么回事
MISSING_RATE_ALERT = 0.10
OFF_GRID_ALERT = 0.001  # 不落采样网格的记录比例


@dataclass(frozen=True)
class Prescription:
    """一条处方。`rule_type` 非空表示能一键生成预处理提案。"""

    text: str
    rule_type: str | None = None
    rule_params: dict[str, Any] = field(default_factory=dict)
    cli_hint: str | None = None

    @property
    def actionable(self) -> bool:
        return self.rule_type is not None and self.rule_type in RULE_LIBRARY


@dataclass(frozen=True)
class Finding:
    """一条诊断结论 + 它的处方。"""

    key: str
    title: str
    severity: Severity
    detail: str
    prescriptions: list[Prescription] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def icon(self) -> str:
        return SEVERITY_ICONS[self.severity]

    @property
    def severity_label(self) -> str:
        return SEVERITY_LABELS[self.severity]


# 五类语义检查的人话说明 + 处方。文案落在这里而不是散在页面上，
# 是为了让「诊断结论」和「该怎么办」永远成对出现。
_CHECK_GUIDE: dict[str, dict[str, Any]] = {
    "derivation_chain": {
        "title": "派生链检查",
        "explain": "候选输入里有没有目标的上游或下游派生量。",
        "why_bad": (
            "用公式推出来的量去预测它的原料，是循环论证。本站点的历史教训："
            "`load = current_percent × 9672 / 100`，拿 load 预测功率能得到"
            "MAPE 4.35% 的漂亮结果，但那只是把同一个电信号换算了两次。"),
        "fix": (
            "从候选输入白名单里移除这些变量。这**不是预处理能解决的**——"
            "清洗改不了「它本来就是目标的换算」这个事实。"),
    },
    "same_origin": {
        "title": "同源相关性",
        "explain": "候选输入与目标的相关性是否高到不像两个独立测点。",
        "why_bad": (
            "相关性接近 1 通常意味着两列来自同一个物理信号的不同刻度，"
            "而不是「这个特征特别有用」。"),
        "fix": (
            "逐个核对高相关变量的测点来源；确认同源就移出白名单。"
            "如果确实是不同测点的强物理耦合，在目标描述里写清理由。"),
    },
    "device_diversity": {
        "title": "设备多样性",
        "explain": "参与建模的对象够不够多、彼此差异够不够大。",
        "why_bad": "只有一台设备时，模型学到的是这台机器的脾气，换台就不准。",
        "fix": "扩大数据范围纳入更多设备；做不到就在模型包约束里限定适用对象。",
    },
    "physical_plausibility": {
        "title": "物理合理性",
        "explain": "COP、温差等衍生量是否落在物理可能范围内。",
        "why_bad": (
            "越界值往往是测点标签错、单位错或公式错。注意范围是参数化的："
            "本站是高温离心式冷机，COP 9~11 物理成立，不能按常规冷机口径"
            "判死（I-48 的教训）。"),
        "fix": "找出越界样本对应的时段与测点，与数据提供方核对标签和量纲。",
    },
    "operating_coverage": {
        "title": "工况覆盖",
        "explain": "有效运行样本够不够、跨越的时间够不够长。",
        "why_bad": "覆盖不足的模型只在见过的工况里成立，外推就是猜。",
        "fix": "延长取数时间窗，或把目标收窄到数据确实覆盖的工况区间。",
    },
}


def from_modelability(report: dict[str, Any]) -> list[Finding]:
    """把 modelability 报告翻译成带处方的诊断条目。"""
    findings: list[Finding] = []
    for check in report.get("checks") or []:
        name = str(check.get("name"))
        guide = _CHECK_GUIDE.get(name, {})
        level = str(check.get("level") or "info")
        severity: Severity = (
            "blocker" if level == "blocker"
            else "warning" if level == "warning" else "info")
        detail_parts = [str(check.get("summary") or "")]
        if severity != "info":
            if guide.get("why_bad"):
                detail_parts.append(f"**为什么要紧**：{guide['why_bad']}")
        elif guide.get("explain"):
            detail_parts.append(guide["explain"])
        prescriptions = []
        if severity != "info" and guide.get("fix"):
            prescriptions.append(Prescription(text=guide["fix"]))
        findings.append(Finding(
            key=name,
            title=guide.get("title", name),
            severity=severity,
            detail="\n\n".join(part for part in detail_parts if part),
            prescriptions=prescriptions,
            evidence=dict(check.get("evidence") or {}),
        ))
    return findings


def from_profile(profile: dict[str, Any]) -> list[Finding]:
    """统计画像里能直接指向处置动作的几件事。"""
    findings: list[Finding] = []

    intervals = profile.get("interval_stats") or {}
    off_grid = float(intervals.get("off_resolution_fraction") or 0.0)
    if off_grid > OFF_GRID_ALERT:
        findings.append(Finding(
            key="time_axis",
            title="时间轴不齐",
            severity="warning",
            detail=(
                f"{off_grid * 100:.2f}% 的记录不落在采样网格上"
                f"（中位间隔 {intervals.get('median_s')}s，"
                f"最大 {intervals.get('max_s')}s）。切分边界要向下取整到采样"
                "周期的整数倍，时间轴歪了会让边界对不上真实样本。"),
            prescriptions=[Prescription(
                text="用 `repair_time_axis` 规则把时间轴重整到统一网格。"
                     "这是登记在规则库里的确定性变换，参数由你给、由人审批，"
                     "不是让 Agent 现写代码。",
                rule_type="repair_time_axis",
                cli_hint="tf --actor human preprocess approve <RULESET>",
            )],
            evidence=dict(intervals),
        ))

    gaps = int(profile.get("gap_count") or 0)
    if gaps:
        findings.append(Finding(
            key="gaps",
            title="时间断档",
            severity="warning",
            detail=(f"存在 {gaps} 处断档。断档本身不一定是问题（停机时段就"
                    "该没数据），但如果断在关键工况上，训练集就缺了那段行为。"),
            prescriptions=[Prescription(
                text="先看断档落在什么时段：停机期可以接受；运行期缺数据要"
                     "回源补，或把该时段排除出 Dataset View 的过滤条件。")],
            evidence={"gap_count": gaps},
        ))

    high_missing = [
        (str(v.get("variable_id")), float(v.get("missing_rate") or 0.0))
        for v in entries(profile.get("variables"))
        if float(v.get("missing_rate") or 0.0) >= MISSING_RATE_ALERT]
    if high_missing:
        high_missing.sort(key=lambda pair: pair[1], reverse=True)
        listing = "、".join(f"`{name}`（{rate * 100:.1f}%）"
                            for name, rate in high_missing[:6])
        findings.append(Finding(
            key="missing",
            title="高缺失率变量",
            severity="warning",
            detail=(f"{len(high_missing)} 个变量缺失率 ≥ "
                    f"{MISSING_RATE_ALERT * 100:.0f}%：{listing}。"
                    "缺失行在建模时会被整行丢弃，缺失率高的变量会连带削掉"
                    "大量本可用的样本。"),
            prescriptions=[Prescription(
                text="权衡：要么把该变量移出特征集换回样本量，要么接受样本"
                     "缩水。**不要在这里做填充**——填充值会被后续当成实测，"
                     "而系统靠 source_kind 区分实测与派生。")],
            evidence={"variables": high_missing},
        ))

    constant = [str(v.get("variable_id"))
                for v in entries(profile.get("variables"))
                if v.get("constant")]
    if constant:
        findings.append(Finding(
            key="constant",
            title="恒定变量",
            severity="info",
            detail=("这些变量全程取值不变，对模型没有信息量："
                    + "、".join(f"`{name}`" for name in constant[:8])),
            prescriptions=[Prescription(text="从候选输入里去掉，减少无谓的维度。")],
            evidence={"variables": constant},
        ))

    violations = int(profile.get("range_violations") or 0)
    if violations:
        findings.append(Finding(
            key="range",
            title="越界取值",
            severity="warning",
            detail=(f"{violations} 个取值超出变量声明的 min/max。越界通常是"
                    "单位错、符号错或测点串了——比如流量出现负值。"),
            prescriptions=[Prescription(
                text="先查是不是量纲/符号问题。确认是坏点再讨论剔除；"
                     "**大面积越界不要当噪声抹掉**，那是数据本身的问题。")],
            evidence={"range_violations": violations},
        ))
    return findings


@dataclass(frozen=True)
class QualityReport:
    verdict: str
    findings: list[Finding]
    target: str | None
    candidate_inputs: list[str]

    @property
    def blockers(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "blocker"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def passed(self) -> bool:
        return not self.blockers

    @property
    def actionable(self) -> list[Finding]:
        return [f for f in self.findings
                if any(p.actionable for p in f.prescriptions)]

    @property
    def headline(self) -> str:
        if self.blockers:
            return (f"{len(self.blockers)} 项阻断，不能进入建模。"
                    "语义门禁 FAIL 会直接挡住实验，先把阻断项处理掉。")
        if self.warnings:
            return (f"没有阻断项，{len(self.warnings)} 项警告。可以开始建模，"
                    "但这些警告会影响结果的可信度，建议先看一眼。")
        return "全部通过，数据可以拿来建模。"


def build_report(modelability: dict[str, Any] | None,
                 profile: dict[str, Any] | None) -> QualityReport:
    """合并两个来源的诊断。语义门禁在前，统计画像在后。"""
    findings: list[Finding] = []
    verdict = "UNKNOWN"
    target = None
    candidate_inputs: list[str] = []
    if modelability:
        findings.extend(from_modelability(modelability))
        verdict = str(modelability.get("verdict") or "UNKNOWN")
        target = modelability.get("target")
        candidate_inputs = list(modelability.get("candidate_inputs") or [])
    if profile:
        findings.extend(from_profile(profile))
    order = {"blocker": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda f: order[f.severity])
    return QualityReport(verdict=verdict, findings=findings, target=target,
                         candidate_inputs=candidate_inputs)
