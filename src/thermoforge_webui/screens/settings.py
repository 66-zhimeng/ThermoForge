"""设置页：模型接口配置 + 分层网络自检。

自检做成逐层是有原因的：「连不上」的成因差别极大——DNS 失败是解析问题，
TCP 失败多半是防火墙，TLS 失败常是中间设备在换证书，而 HTTP 通了就说明
网络没问题、该去看密钥和模型名。一句「连接失败」没法照着做。
"""

from __future__ import annotations

import streamlit as st

from .. import cache
from ..config import PROVIDERS, check_endpoint, diagnose_network, read_config, write_config
from ..ui import copilot_banner


def settings_page() -> None:
    st.title("设置")
    copilot_banner("settings")
    config = read_config()

    if config.from_env:
        st.info("检测到环境变量 `TF_AGENT_API_KEY`，它的优先级高于配置文件。"
                "在这里保存的密钥不会生效，除非先清掉环境变量。", icon="🌱")

    left, right = st.columns([3, 2], gap="large")
    with left:
        _form(config)
    with right:
        _diagnostics()


def _form(config) -> None:
    st.subheader("模型接口")
    st.caption("任何 OpenAI 兼容的 chat completions + function calling 端点都能用，"
               "包括自建网关（one-api / new-api / vLLM / LM Studio）。"
               "下面的服务商目录只是把常见地址填对，不构成限制。")

    names = {p["id"]: p["name"] for p in PROVIDERS}
    provider_id = st.selectbox("服务商（仅用于填默认值）", options=list(names),
                               format_func=lambda k: names[k],
                               key="settings_provider")
    provider = next(p for p in PROVIDERS if p["id"] == provider_id)
    if provider.get("note"):
        st.caption(f"💡 {provider['note']}")

    with st.form("agent_config"):
        api_key = st.text_input(
            "API 密钥", type="password",
            placeholder=(f"当前 {config.api_key_masked}（留空则保留不变）"
                         if config.configured else "粘贴你的密钥"))
        base_url = st.text_input("接口地址",
                                 value=config.base_url or provider["base_url"])
        model_options = provider.get("models") or []
        model = st.text_input(
            "模型名", value=config.model,
            help=("该服务商常见模型：" + "、".join(model_options))
            if model_options else "以服务商文档为准")
        proxy = st.text_input(
            "代理（可选）", value=config.proxy,
            placeholder="http://127.0.0.1:7890",
            help="公司网络或需要翻墙时填。留空表示直连。")
        submitted = st.form_submit_button("保存", type="primary")

    if submitted:
        try:
            write_config(api_key.strip(), base_url.strip(), model.strip(),
                         proxy.strip())
        except ValueError as exc:
            st.error(str(exc))
        else:
            cache.invalidate()
            st.session_state.pop("research_session", None)  # 配置变了，旧会话作废
            st.success("已保存。密钥写入 harness/agent.toml（该文件已 gitignore）。")
            st.rerun()

    st.caption(f"配置文件：`{config.config_path}`　密钥读回一律掩码，"
               "完整值不会出现在界面或日志里。")


def _diagnostics() -> None:
    st.subheader("连通性自检")
    columns = st.columns(2)
    if columns[0].button("快速检查", width="stretch"):
        with st.spinner("正在调用模型接口…"):
            result = check_endpoint()
        if result.get("ok"):
            st.success(f"通了：模型 `{result.get('model')}` 响应正常。")
        else:
            st.error(result.get("error") or "失败")
            if result.get("detail"):
                st.caption(f"原始错误：`{result['detail']}`")

    if columns[1].button("分层排查", width="stretch", type="primary"):
        with st.spinner("DNS → TCP → TLS → HTTP…"):
            result = diagnose_network()
        st.caption(f"目标 `{result.get('host')}:{result.get('port')}`")
        for step in result.get("steps") or []:
            icon = "✅" if step.get("ok") else "❌"
            duration = f"{step['ms']}ms" if step.get("ms") is not None else ""
            st.markdown(f"{icon} **{step['name']}**　`{duration}`")
            st.caption(step.get("detail") or "")
        verdict = result.get("verdict")
        if verdict:
            (st.success if result.get("ok") else st.warning)(verdict)
