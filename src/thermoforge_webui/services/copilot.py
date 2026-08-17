"""网页副驾：听懂问题 → 查数据 → 给结论 → 把你带到对应页面。

复用 `HarnessAgent` 的对话循环（function calling + 审批拦截 + 会话留痕），在它
的工具表上**追加一个界面工具** `ui_goto`——模型除了能查数据，还能决定
「这个问题该看哪一页、该选中哪个实验」，并把结论一起带过去。

追加方式是往实例的 `schemas` / `dispatch` 上加，不改 `thermoforge_agent`：
界面能力属于界面，不该渗进 CLI 和 MCP 共用的 Agent 里。

一轮问答跑在后台线程里：副驾有权跑实验（使用者选的「全自动，除了
human-only 的」），一次 `tf_experiment_run` 可能要几十秒到几分钟，
放在 Streamlit 脚本线程里会把整个界面冻住。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..context import tool_context
from ..navigation import PAGE_KEYS, catalog_for_prompt

# 人工审批的等待上限：超时就当拒绝，别让后台线程永远挂着
APPROVAL_TIMEOUT_SECONDS = 900.0

UI_GOTO_TOOL = "ui_goto"

UI_GOTO_SCHEMA = {
    "type": "function",
    "function": {
        "name": UI_GOTO_TOOL,
        "description": (
            "把使用者带到控制台的某个页面，并预先选好要看的对象。"
            "只要问题的答案配合界面上的图表更容易讲清楚，就调用它——"
            "使用者明确说了不想自己点来点去。把你的结论写进 note，"
            "它会显示在目标页顶部。一轮对话最多调用一次。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "page": {"type": "string", "enum": list(PAGE_KEYS),
                         "description": "目标页面 key"},
                "note": {"type": "string",
                         "description": "显示在目标页顶部的结论（中文，2~4 句，"
                                        "说清看点在哪，不要复述页面上已有的数字）"},
                "experiment_id": {"type": "string",
                                  "description": "results 页要选中的实验，如 EXP-0013"},
                "surface": {"type": "string",
                            "description": "results 页要看的评估面：train/validate/A/B/C"},
                "dataset_ref": {"type": "string",
                                "description": "data / quality 页要选中的数据集，"
                                               "如 WX_2025_PLANT@rev_0001"},
                "target": {"type": "string",
                           "description": "quality 页体检的目标变量属性码"},
            },
            "required": ["page", "note"],
        },
    },
}

# 文件缺失时的兜底（内容与 copilot.md 保持同义；真正生效的是那个文件）
_FALLBACK_PROMPT = f"""你是 ThermoForge 控制台的副驾。ThermoForge 是数据中心暖通的
物理-数据混合建模研究系统，使用者是工程师，但不想自己在界面上点来点去——
他提问，你负责查清楚、讲明白，并把他带到该看的页面。

## 你的工作方式

1. **先查再答**。所有结论都要有工具查到的数据支撑，不要凭印象说。
2. **答完就带路**。只要配合图表更好讲，就调用 {UI_GOTO_TOOL} 跳到对应页面并选好对象。
3. **说人话**。指标是比率不是百分数（CVRMSE 0.116 就是 11.6%）；R² 可以是负的，
   负值表示这个模型比直接取平均还差，这是有信息量的结论，不要藏着。
4. **不确定就说不确定**，不要编 ID、不要编数字。

## 控制台页面

{catalog_for_prompt()}

## 这个系统的几条硬规矩（回答时要守住）

- **候选输入白名单是硬门禁**：由公式算出来的派生量不能用来预测它的原料，
  那是循环论证。本站的历史教训是 `load = current_percent × 9672 / 100`，
  拿 load 预测功率能得到很漂亮的 MAPE，但毫无意义。
- **工具返回信封，失败也返回**：必须看 `ok` 字段，不是看有没有报错。
- **指标只有一份实现**，你不要自己算 RMSE/CVRMSE，读工具给的。
- **内置模型路线是闭集，不要试探名称**：data 只支持 `ridge`/`linear`；physics
  支持 `cooling_balance_v1`/`cooling_balance_v2` 与 `gordon_ng`/`eps_ntu`；
  hybrid 使用上述 physics
  加 `residual=xgboost`。当前没有 MLP、神经网络、LightGBM 或纯 data XGBoost；
  用户要求未实现路线时直接说明能力缺口，不要反复调用实验工具猜 estimator。
- **闭集之外的新模型走模型实验室，且这条路你自己走完**：用 tf_lab_submit
  提交单文件模型代码（协议见 tf_lab_list/tf_lab_get 返回的说明与校验报告），
  静态扫描 + 结构校验一过就是 validated，**不需要任何人审批**，立刻能以
  category=lab + hyperparameters.lab 开实验跑真实数据。看完指标要改模型
  就改源码重交（自动进新版本），走不通的方案用 tf_lab_deprecate 停掉。
- **预处理审批必须由人来点**。需要审批时用 tf_human_approval 发起，
  界面会弹给使用者确认，你不能替他批。
- 实验按时间切分、子进程隔离执行、种子固定，同机重跑指标应逐位一致。
"""


def system_prompt() -> str:
    """副驾提示词：以 `harness/prompts/copilot.md` 为准，`_FALLBACK_PROMPT` 兜底。

    每次建 agent 时现读（不在 import 期定死）：改提示词文件应当立刻生效，
    否则「装技能 = 改文件」就变成「改完还得重启界面」。页面清单与界面
    工具名是代码派生的事实，以占位符填进去。
    """
    from thermoforge_agent import prompts

    return prompts.load(
        "copilot",
        fallback=_FALLBACK_PROMPT,
        page_catalog=catalog_for_prompt(),
        ui_goto_tool=UI_GOTO_TOOL,
    )


@dataclass
class CopilotEvent:
    kind: str  # tool / approval / answer / error
    text: str
    at: float = field(default_factory=time.time)
    ok: bool = True


@dataclass
class Message:
    role: str  # user / assistant
    content: str
    at: float = field(default_factory=time.time)


class CopilotSession:
    """一次浏览器会话里的副驾。线程安全靠一把锁 + 一个审批闸门。"""

    def __init__(self, client: Any | None = None) -> None:
        # client 只为测试注入 stub；生产走 HarnessAgent 内部的 ChatClient，不触网测不了
        self._client = client
        self._lock = threading.Lock()
        self._agent: Any | None = None
        self._thread: threading.Thread | None = None
        self._decision = threading.Event()
        self._approved = False

        self.messages: list[Message] = []
        self.events: list[CopilotEvent] = []
        self.state = "idle"  # idle / thinking / awaiting_approval / error
        self.error: str | None = None
        self.pending_approval: dict[str, Any] | None = None
        self.nav_intent: dict[str, Any] | None = None

    # ---- 状态

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def snapshot(self) -> tuple[list[Message], list[CopilotEvent], str]:
        with self._lock:
            return list(self.messages), list(self.events), self.state

    def reset(self) -> None:
        if self.busy:
            return
        with self._lock:
            self._agent = None
            self.messages.clear()
            self.events.clear()
            self.state = "idle"
            self.error = None
            self.nav_intent = None

    # ---- 提问

    def ask(self, question: str, config: Any) -> bool:
        """发起一轮问答（后台线程）。返回是否真的起了新一轮。"""
        if self.busy or not question.strip():
            return False
        with self._lock:
            self.messages.append(Message("user", question.strip()))
            self.events.clear()
            self.nav_intent = None
            self.state = "thinking"
            self.error = None
        self._thread = threading.Thread(target=self._run, args=(question, config),
                                        name="copilot", daemon=True)
        self._thread.start()
        return True

    def _run(self, question: str, config: Any) -> None:
        try:
            agent = self._ensure_agent(config)
            answer = agent.ask(question)
        except Exception as exc:
            with self._lock:
                self.error = f"{type(exc).__name__}: {exc}"
                self.state = "error"
                self.events.append(CopilotEvent("error", self.error, ok=False))
            return
        with self._lock:
            self.messages.append(Message("assistant", answer))
            self.events.append(CopilotEvent("answer", "已给出回答"))
            self.state = "idle"

    # ---- Agent 装配

    def _ensure_agent(self, config: Any):
        from thermoforge_agent import HarnessAgent

        if self._agent is not None:
            return self._agent
        agent = HarnessAgent(
            config, tool_context(),
            client=self._client,
            system_prompt=system_prompt(),
            approval_handler=self._on_approval,
            on_tool_call=self._on_tool_call,
        )
        # 追加界面工具：模型除了查数据，还能决定「该看哪一页」
        agent.schemas = [*agent.schemas, UI_GOTO_SCHEMA]
        agent.dispatch = {**agent.dispatch, UI_GOTO_TOOL: self._ui_goto}
        self._agent = agent
        return agent

    # ---- 工具回调（都跑在后台线程里，不能碰 st.*）

    def _on_tool_call(self, tool: str, ok: bool) -> None:
        with self._lock:
            self.events.append(
                CopilotEvent("tool", tool, ok=ok))

    def _ui_goto(self, _ctx: Any, page: str, note: str = "",
                 **selections: Any) -> dict[str, Any]:
        """界面工具：记下跳转意图，由主流程执行（这里没有脚本上下文）。"""
        if page not in PAGE_KEYS:
            return _envelope(UI_GOTO_TOOL, False,
                             {"error": f"未知页面 {page!r}，可选 {list(PAGE_KEYS)}"})
        with self._lock:
            self.nav_intent = {"page": page, "note": note,
                               "selections": dict(selections)}
            self.events.append(CopilotEvent("nav", f"跳转到「{page}」"))
        return _envelope(UI_GOTO_TOOL, True,
                         {"navigated_to": page,
                          "note": "已记录跳转，使用者会在回答给出后被带过去"})

    def _on_approval(self, tool: str, arguments: Mapping[str, Any],
                     reason: str) -> bool:
        """human-only 工具：卡住后台线程，等使用者在界面上点。"""
        with self._lock:
            self.pending_approval = {"tool": tool, "arguments": dict(arguments),
                                     "reason": reason}
            self.state = "awaiting_approval"
            self.events.append(CopilotEvent("approval", f"请求审批 {tool}"))
        self._decision.clear()
        got = self._decision.wait(timeout=APPROVAL_TIMEOUT_SECONDS)
        with self._lock:
            self.pending_approval = None
            self.state = "thinking"
            approved = bool(got and self._approved)
            self.events.append(CopilotEvent(
                "approval", "已批准" if approved else "已拒绝（或超时）",
                ok=approved))
        return approved

    def decide_approval(self, approved: bool) -> None:
        self._approved = approved
        self._decision.set()


def _envelope(tool: str, ok: bool, summary: dict[str, Any]) -> dict[str, Any]:
    """界面工具也走信封格式，模型看到的结构与其它工具一致。"""
    return {"ok": ok, "tool": tool, "id": None,
            "status": "OK" if ok else "FAILED", "inputs": {},
            "summary": summary, "diagnostics": [], "artifacts": [],
            "truncated": False}
