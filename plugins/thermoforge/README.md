# ThermoForge 研究副驾驶插件

这个插件是独立 ThermoForge 软件的操作入口，提供 `tf_v2_*` MCP 工具与 `thermoforge-control` 技能。软件负责后台研究、六个 Codex 实例、实验队列和报告；插件关闭后，后台服务仍按已授权配置运行。

## 本地准备

软件目录需要已经完成 `uv sync`，Codex 由软件自身检查安装与认证。插件引导脚本只使用 Python 标准库，要求 `python` 可从 PATH 启动；真正的 MCP 运行在软件的 `.venv` 中。

从仓库内直接运行时，启动器会沿插件父目录发现 ThermoForge。先做无模型调用的路径检查：

```powershell
python plugins/thermoforge/scripts/launch_mcp.py --check
```

插件复制到安装缓存后不再位于软件仓库下。分发或安装前，保存这台机器的软件位置：

```powershell
python plugins/thermoforge/scripts/configure.py --project-root "D:/path/to/ThermoForge"
```

这只创建插件目录内的 `connection.json`，不修改个人插件市场、全局 Codex 配置或认证。该文件包含本机路径，已在插件的 `.gitignore` 中排除；跨机器分发时应移除它，由接收者配置自己的路径。也可向插件进程提供 `TF_PROJECT_ROOT`，其优先级高于连接文件。

`.mcp.json` 使用 `cwd: "."` 将脚本路径解析到插件根；没有嵌入开发者机器的绝对路径，也没有依赖 `${CLAUDE_PLUGIN_ROOT}` 的变量替换。该行为依据 [Codex 0.151.0 插件配置解析源码](https://github.com/openai/codex/blob/rust-v0.151.0/codex-rs/codex-mcp/src/plugin_config.rs)。

## 交付与安装边界

仓库交付的是插件目录，未自动安装到个人环境，也未创建个人 marketplace。请通过目标客户端支持的本地插件导入流程导入本目录，并确保配置后的 `connection.json` 随本机安装副本保留。

若只需要接入 MCP 而不安装插件，可把以下服务配置交给已有 MCP 客户端，并将路径替换为真实软件目录：

```json
{
  "mcpServers": {
    "thermoforge-v2": {
      "command": "D:/path/to/ThermoForge/.venv/Scripts/python.exe",
      "args": ["-m", "thermoforge_v2.mcp"],
      "cwd": "D:/path/to/ThermoForge"
    }
  }
}
```

Linux/macOS 的解释器路径为 `.venv/bin/python`。这个服务是操作适配器，不是内部研究智能体。实际软件用法见仓库的 `docs/v2-getting-started.md`。
