"""真实子进程上的 App Server 协议测试；不调用模型、不消耗 token。"""
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

from thermoforge_v2.codex import (
    CodexConfig, CodexError, CodexSession, DISABLED_FEATURES, PERMISSION_PROFILE,
    restricted_config,
)


FAKE_SERVER = r'''
import json,os,sys
def out(d): print(json.dumps(d),flush=True)
def event(m,p): out({'method':m,'params':p})
def done(status='completed'):
 event('item/completed',{'threadId':thread,'item':{'type':'agentMessage','id':'reply','phase':'final_answer','text':'已完成研究'}})
 event('turn/completed',{'threadId':thread,'turn':{'id':turn,'status':status,'items':[],'error':None}})
config={}
for i,a in enumerate(sys.argv):
 if a=='-c':
  import tomllib
  k,v=sys.argv[i+1].split('=',1)
  config[k]=tomllib.loads('value='+v)['value']
thread='thread-'+str(os.getpid());turn=None;pending=0
for line in sys.stdin:
 m=json.loads(line);method=m.get('method');p=m.get('params',{});i=m.get('id')
 if method=='initialize':r={'userAgent':'Codex Desktop/0.153.4 test'}
 elif method=='initialized':continue
 elif method=='account/read':r={'requiresOpenaiAuth':True,'account':None if os.environ.get('FAKE_UNAUTH') else {'type':'chatgpt'}}
 elif method=='config/read':
  profile={'filesystem':config['permissions.thermoforge_research.filesystem'],'network':{'enabled':False}}
  if os.environ.get('FAKE_BROADENED'):profile['filesystem']['/unsafe']='read'
  r={'config':{'features':{k.removeprefix('features.'):v for k,v in config.items() if k.startswith('features.')},'mcp_servers':{'inherited':{'enabled':True}},'permissions':{'thermoforge_research':profile}}}
 elif method in ('thread/start','thread/resume'):
  assert p['config']['mcp_servers.inherited.enabled'] is False
  assert p['permissions']=='thermoforge_research'
  assert p['runtimeWorkspaceRoots']==[p['cwd']]
  if method=='thread/resume':thread=p['threadId'];assert 'dynamicTools' not in p
  else:
   assert all(t['type']=='function' for t in p['dynamicTools'])
   assert p['allowProviderModelFallback'] is False
  r={'thread':{'id':thread},'activePermissionProfile':{'id':'thermoforge_research'},'model':'configured-model','modelProvider':'openai'}
 elif method=='turn/start':
  turn='turn-'+str(i);mode=p['input'][0]['text']
  if mode=='disconnect':sys.exit(3)
  event('turn/started',{'threadId':thread,'turn':{'id':turn,'status':'inProgress'}})
  out({'id':i,'result':{'turn':{'id':turn,'status':'inProgress'}}})
  event('thread/tokenUsage/updated',{'threadId':thread,'turnId':turn,'tokenUsage':{'total':{'totalTokens':42},'last':{'totalTokens':21}}})
  event('item/completed',{'threadId':thread,'item':{'type':'reasoning','id':'thought','summary':['摘要'],'content':['不持久化内部推理']}})
  if mode=='wait':continue
  if mode in ('tool','duplicate','wrong-thread','tool-error'):
   pending=2 if mode=='duplicate' else 1
   for n in range(pending):out({'id':100+n,'method':'item/tool/call','params':{'tool':'tf_probe','arguments':{'value':7},'callId':'call-'+turn,'threadId':'other' if mode=='wrong-thread' else thread,'turnId':turn}})
  else:done()
  continue
 elif method=='turn/interrupt':
  out({'id':i,'result':{}});done('interrupted');continue
 elif method is None:
  if i in (100,101):
   result=m['result'];assert result['contentItems'][0]['type']=='inputText'
   payload=json.loads(result['contentItems'][0]['text'])
   if mode in ('wrong-thread','tool-error'):assert result['success'] is False
   else:assert payload['value']==7 and result['success'] is True
   pending-=1
   if not pending:done()
  continue
 else:out({'id':i,'error':{'code':-32601,'message':'unknown'}});continue
 out({'id':i,'result':r})
'''


@pytest.fixture
def config(tmp_path):
    script = tmp_path / "fake_server.py"
    script.write_text(FAKE_SERVER, encoding="utf-8", newline="\n")
    return CodexConfig(cwd=tmp_path / "workspace", command=[sys.executable, "-u", str(script)],
                       request_timeout=3, turn_timeout=5, shutdown_timeout=1)


SPECS = [{"name": "tf_probe", "description": "协议测试工具", "inputSchema": {"type": "object", "properties": {"value": {"type": "integer"}}}}]


async def echo(name, args):
    return {"ok": True, **args}


def test_independent_processes_tools_and_resume(config):
    async def scenario():
        sessions = [CodexSession(replace(config, cwd=config.cwd / str(n)), SPECS, echo) for n in range(6)]
        try:
            infos = await asyncio.gather(*(s.start() for s in sessions))
            assert len({s.pid for s in sessions}) == 6
            assert len({s.thread_id for s in sessions}) == 6
            assert all(i["native_subagents"] is False for i in infos)
            results = await asyncio.gather(*(s.turn("tool") for s in sessions))
            assert all(r.status == "completed" and r.text == "已完成研究" for r in results)
            assert all(r.usage["total"]["totalTokens"] == 42 for r in results)
            thread = sessions[0].thread_id
            old_pid = sessions[0].pid
        finally:
            await asyncio.gather(*(s.close() for s in sessions))
        resumed = CodexSession(config, SPECS, echo)
        try:
            await resumed.start(thread)
            assert resumed.thread_id == thread and resumed.pid != old_pid
            assert (await resumed.turn("done")).status == "completed"
            assert (await resumed.turn("done")).status == "completed"
        finally:
            await resumed.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["duplicate", "wrong-thread", "tool-error"])
def test_tool_idempotency_context_and_failure(config, mode):
    async def scenario():
        calls = []
        async def tool(name, args):
            calls.append(args)
            await asyncio.sleep(0.02)
            if mode == "tool-error":
                raise ValueError("工具故障")
            return {"ok": True, **args}
        session = CodexSession(config, SPECS, tool)
        try:
            await session.start()
            assert (await session.turn(mode)).status == "completed"
            assert len(calls) == (0 if mode == "wrong-thread" else 1)
        finally:
            await session.close()
    asyncio.run(scenario())


def test_interrupt_does_not_block_stdio_and_events_omit_raw_reasoning(config):
    async def scenario():
        events = []
        ready = asyncio.Event()
        async def handler(event):
            events.append(event)
            if event["method"] == "turn/started":
                ready.set()
        session = CodexSession(config, SPECS, echo, handler)
        try:
            await session.start()
            task = asyncio.create_task(session.turn("wait"))
            await asyncio.wait_for(ready.wait(), 3)
            with pytest.raises(CodexError):
                await session.turn("done")
            await session.interrupt()
            result = await task
            assert result.status == "interrupted"
            assert session.state == "idle"
        finally:
            await session.close()
        assert "不持久化内部推理" not in json.dumps(events, ensure_ascii=False)
    asyncio.run(scenario())


def test_disconnection_fails_turn_and_closes_process(config):
    async def scenario():
        session = CodexSession(config, SPECS, echo)
        await session.start()
        with pytest.raises(CodexError, match="断开"):
            await session.turn("disconnect")
        assert session.process.returncode is not None
        assert session.state == "closed"
    asyncio.run(scenario())


def test_turn_timeout_really_interrupts_backend(config):
    async def scenario():
        session = CodexSession(replace(config, turn_timeout=0.1), SPECS, echo)
        try:
            await session.start()
            with pytest.raises(TimeoutError):
                await session.turn("wait")
            assert session.state == "idle"
            assert (await session.turn("done")).status == "completed"
        finally:
            await session.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("change,match", [({"env": {"FAKE_UNAUTH": "1"}}, "登录"), ({"expected_version": "0.999.0"}, "版本"), ({"env": {"FAKE_BROADENED": "1"}}, "扩宽")])
def test_preflight_fails_closed(config, change, match):
    async def scenario():
        session = CodexSession(replace(config, **change), SPECS, echo)
        with pytest.raises(CodexError, match=match):
            await session.start()
        assert session.process.returncode is not None
        assert session.state == "closed"
    asyncio.run(scenario())


def test_restricted_config_does_not_grant_root_or_spawn_agents():
    cfg = restricted_config()
    assert all(cfg[f"features.{feature}"] is False for feature in DISABLED_FEATURES)
    fs = cfg[f"permissions.{PERMISSION_PROFILE}.filesystem"]
    assert fs[":root"] == "deny"
    assert fs[":workspace_roots"]["."] == "read"
    assert cfg[f"permissions.{PERMISSION_PROFILE}.network.enabled"] is False
    assert cfg["features.code_mode_host"] is True


def test_event_recording_failure_stops_generation(config):
    async def scenario():
        async def handler(event):
            if event["method"] == "turn/started":
                raise OSError("磁盘不可写")
        session = CodexSession(config, SPECS, echo, handler)
        try:
            await session.start()
            with pytest.raises(CodexError, match="回调失败"):
                await session.turn("wait")
            assert session.process.returncode is not None
        finally:
            await session.close()
    asyncio.run(scenario())


def test_explicit_binary_wins_over_installed_desktop(monkeypatch, tmp_path):
    import thermoforge_v2.codex as module
    explicit = tmp_path / "selected.exe"
    monkeypatch.setenv("TF_CODEX_BIN", str(explicit))
    monkeypatch.setattr(module, "_desktop_codex_command", lambda: ["desktop.exe"])
    assert module.resolve_codex_command() == [str(explicit)]


def test_validated_desktop_runtime_wins_over_old_global_cli(monkeypatch):
    import thermoforge_v2.codex as module
    monkeypatch.delenv("TF_CODEX_BIN", raising=False)
    monkeypatch.setattr(module, "_desktop_codex_command", lambda: ["verified-desktop.exe"])
    monkeypatch.setattr(module.shutil, "which", lambda _: "old-global.exe")
    assert module.resolve_codex_command() == ["verified-desktop.exe"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows 桌面运行时发现")
def test_desktop_discovery_only_accepts_verified_versions(monkeypatch, tmp_path):
    import thermoforge_v2.codex as module
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    root = tmp_path / "OpenAI" / "Codex" / "bin"
    older, verified = root / "old" / "codex.exe", root / "verified" / "codex.exe"
    for path in (older, verified):
        path.parent.mkdir(parents=True)
        path.touch()
    monkeypatch.setattr(module, "_runtime_version", lambda path: "0.151.0" if path == older else module.VALIDATED_CODEX_VERSION)
    assert module._desktop_codex_command() == [str(verified)]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows 权限 helper 初始化")
def test_windows_serializes_only_thread_initialization(config, monkeypatch):
    original = CodexSession._request
    active, peak = 0, 0
    async def measured(self, method, params):
        nonlocal active, peak
        if method not in {"thread/start", "thread/resume"}:
            return await original(self, method, params)
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.01)
            return await original(self, method, params)
        finally:
            active -= 1
    monkeypatch.setattr(CodexSession, "_request", measured)
    async def scenario():
        sessions = [CodexSession(replace(config, cwd=config.cwd / str(i)), SPECS, echo) for i in range(6)]
        try:
            await asyncio.gather(*(s.start() for s in sessions))
            assert peak == 1
            assert len({s.pid for s in sessions}) == 6
            results = await asyncio.gather(*(s.turn("tool") for s in sessions))
            assert all(r.status == "completed" for r in results)
        finally:
            await asyncio.gather(*(s.close() for s in sessions))
    asyncio.run(scenario())
