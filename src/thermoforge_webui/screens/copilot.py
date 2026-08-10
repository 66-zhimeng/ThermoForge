"""AI 助手页：提问 → 副驾查数据 → 给结论 → 把你带到该看的页面。

这一页是「我不想自己点」的入口。副驾有全部 22 个工具的权限（能查数据、
做体检、建目标、跑实验、比模型、发布），唯一拦下来的是需要 `actor=human`
的预处理审批——那一步会弹出来让你亲自点，因为「规则是谁批的」必须留痕。
"""

from __future__ import annotations

import time

import streamlit as st

from .. import cache, navigation
from ..config import agent_config
from ..services.copilot import CopilotSession
from ..ui import empty_state

SESSION_KEY = "copilot_session"
PENDING_QUESTION_KEY = "copilot_pending_question"

# 侧栏快速提问会把问题塞进 session_state，切到本页后自动发出
SUGGESTIONS = [
    "现在整体进展怎么样？",
    "哪次实验效果最好？为什么？",
    "我的数据能拿来建模吗？有什么问题？",
    "最近一次实验为什么这么差？",
]


def get_session() -> CopilotSession:
    session = st.session_state.get(SESSION_KEY)
    if session is None:
        session = CopilotSession()
        st.session_state[SESSION_KEY] = session
    return session


def copilot_page() -> None:
    st.title("AI 助手")
    session = get_session()
    config = agent_config()

    if config is None:
        empty_state("还没配置模型接口",
                    "副驾要调用大模型才能理解问题。去「设置」页填 API 密钥，"
                    "任何 OpenAI 兼容的接口都行。", icon="🔑")
        return

    _header(session)
    _history(session)

    if session.state == "awaiting_approval":
        _approval(session)

    if session.busy or session.state == "thinking":
        _progress(session)
    else:
        _finish_turn(session)

    question = st.chat_input("问点什么，比如「哪次实验最好，为什么」",
                             disabled=session.busy)
    pending = st.session_state.pop(PENDING_QUESTION_KEY, None)
    asked = question or pending
    if asked:
        session.ask(str(asked), config)
        st.rerun()

    if not session.messages and not session.busy:
        _suggestions(session, config)


def _header(session: CopilotSession) -> None:
    columns = st.columns([5, 1])
    columns[0].caption(
        "副驾能查数据、做体检、跑实验、比模型，并把你带到对应页面。"
        "需要人工审批的动作（预处理规则）会停下来让你点。")
    if columns[1].button("清空对话", width="stretch", disabled=session.busy):
        session.reset()
        st.rerun()


def _history(session: CopilotSession) -> None:
    messages, _, _ = session.snapshot()
    for message in messages:
        with st.chat_message("user" if message.role == "user" else "assistant"):
            st.markdown(message.content)


@st.fragment(run_every=1.5)
def _progress(session: CopilotSession) -> None:
    """思考中的实时进度。副驾可能连着调好几个工具，得让人看见它在干活。"""
    _, events, state = session.snapshot()
    with st.status("正在查…", expanded=True):
        for event in events:
            stamp = time.strftime("%H:%M:%S", time.localtime(event.at))
            icon = {"tool": "🔧", "nav": "🧭", "approval": "🙋",
                    "error": "❌", "answer": "✅"}.get(event.kind, "•")
            mark = "" if event.ok else "（失败）"
            st.write(f"{icon} `{stamp}` {event.text}{mark}")
        if not events:
            st.write("🤔 正在想…")
    if not session.busy and state != "awaiting_approval":
        st.rerun(scope="app")  # 跑完了，回主流程去处理跳转


def _finish_turn(session: CopilotSession) -> None:
    """一轮结束后的收尾：报错、或者按副驾的意思跳页。"""
    if session.error:
        st.error(f"副驾出错了：{session.error}")
        session.error = None
        return
    intent = session.nav_intent
    if not intent:
        return
    session.nav_intent = None
    cache.invalidate()  # 副驾可能刚跑过实验，目标页要读到新工件
    if not navigation.apply_and_switch(intent, validator=_valid_selection):
        st.warning(f"副驾想跳到「{intent.get('page')}」，但那个页面没注册。")


def _valid_selection(name: str, value: object) -> bool:
    """预选值要先验证还在：给 selectbox 塞不存在的选项会直接抛异常。"""
    if name == "results_exp":
        return any(item.experiment_id == value
                   for item in cache.experiment_list())
    if name in ("data_ref", "quality_ref"):
        return any(row.ref == value for row in cache.revisions())
    return True


def _approval(session: CopilotSession) -> None:
    request = session.pending_approval
    if not request:
        return
    st.warning("副驾请求执行一个**必须由人批准**的动作。", icon="🙋")
    with st.container(border=True):
        st.markdown(f"**工具**　`{request['tool']}`")
        st.markdown(f"**理由**　{request.get('reason') or '（未说明）'}")
        st.json(request.get("arguments") or {})
        st.caption("批准后会以 `actor=human` 执行，Ledger 里记的是你批的——"
                   "这是预处理规则审批的留痕要求，副驾代替不了。")
        columns = st.columns(2)
        if columns[0].button("批准执行", type="primary", width="stretch"):
            session.decide_approval(True)
            st.rerun()
        if columns[1].button("拒绝", width="stretch"):
            session.decide_approval(False)
            st.rerun()


def _suggestions(session: CopilotSession, config) -> None:
    st.caption("不知道问什么？点一个：")
    columns = st.columns(2)
    for index, text in enumerate(SUGGESTIONS):
        if columns[index % 2].button(text, width="stretch",
                                     key=f"suggest_{index}"):
            session.ask(text, config)
            st.rerun()
