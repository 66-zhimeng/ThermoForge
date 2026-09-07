"""Offline, inspectable research flow for exported reports; no model calls or CDN."""

from __future__ import annotations

from html import escape
import json
import math
from typing import Any, Mapping
from urllib.parse import urlsplit


def _safe_data(value: Any, key: str = "") -> Any:
    if isinstance(value, Mapping):
        return {str(k): _safe_data(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_data(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if key == "url" and isinstance(value, str):
        try:
            parsed = urlsplit(value)
            return value if parsed.scheme in {"http", "https"} and parsed.netloc else ""
        except ValueError:
            return ""
    return value


def render_flow_html(flow: Mapping[str, Any] | None) -> str:
    if not flow or not flow.get("nodes"):
        return ""
    payload = json.dumps(_safe_data(flow), ensure_ascii=False, default=str, allow_nan=False)
    # A JSON script element is still parsed by the HTML tokenizer before JSON.parse.
    for char, replacement in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                              ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        payload = payload.replace(char, replacement)
    notes = " ".join(str(note) for note in flow.get("notes") or [])
    return ("<style>" + _CSS + "</style>" + _MARKUP.replace("FLOW_NOTES", escape(notes))
            + '<script type="application/json" id="tf-flow-data">' + payload + "</script>"
            + '<script type="module">' + _JS + "</script>")


_MARKUP = """
<section id="tf-flow" aria-label="研究流程追踪">
  <div class="tf-heading"><div><span class="tf-eyebrow">RESEARCH TRACE</span>
    <h2>从想法到实验，再到结论</h2></div><span class="tf-summary"></span></div>
  <p class="tf-intro">每行是一条独立研究轨迹。点击节点查看研究理由、实测指标和引用证据；图中展示主要关系，其余引用见节点详情。位置不代表精确执行时间。</p>
  <div class="tf-toolbar"><label>研究轨迹 <select class="tf-track" aria-label="选择研究轨迹"><option value="all">全部轨迹</option></select></label>
    <label>证据记录 <select class="tf-record" aria-label="直接查看登记记录"><option value="">选择来源、发现或历史报告</option></select></label>
    <span class="tf-legend"><i></i>来源 / 实验依据 <i class="tf-dashed"></i>任务传递 / 记录归属</span></div>
  <div class="tf-workspace">
    <div class="tf-map-scroll" tabindex="0" aria-label="研究流程图，可横向滚动">
      <div class="tf-map"><svg class="tf-edges" aria-hidden="true"></svg><div class="tf-lanes"></div></div>
    </div>
    <aside class="tf-detail" aria-label="节点详情" aria-live="polite"><p>选择一个节点查看证据。</p></aside>
  </div>
  <p class="tf-footnote">FLOW_NOTES</p>
  <noscript>流程图需要启用 JavaScript；下方完整文字报告保留全部研究事实。</noscript>
</section>
"""

_CSS = """
#tf-flow{--tf-ink:#203039;--tf-muted:#536775;--tf-line:#d8e1e5;--tf-green:#176553;--tf-wash:#edf5f2;color:var(--tf-ink);margin:30px 0 34px}
#tf-flow *{box-sizing:border-box}#tf-flow .tf-heading{display:flex;justify-content:space-between;align-items:center;gap:20px}
#tf-flow h2{border:0;padding:0;margin:5px 0 8px;font-size:25px}#tf-flow .tf-eyebrow{font-size:11px;letter-spacing:.15em;color:var(--tf-green);font-weight:600}
#tf-flow .tf-summary{font-size:13px;color:var(--tf-muted);white-space:nowrap}#tf-flow .tf-intro,#tf-flow .tf-footnote{font-size:13px;color:var(--tf-muted);line-height:1.7}
#tf-flow .tf-toolbar{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;margin:18px 0}
#tf-flow label{font-size:13px}#tf-flow select{font:inherit;color:inherit;border:1px solid var(--tf-line);border-radius:6px;padding:7px 12px;background:white;margin-left:8px}
#tf-flow .tf-record{max-width:290px}#tf-flow .tf-toolbar label{max-width:100%}
#tf-flow .tf-legend{font-size:12px;color:var(--tf-muted);display:flex;align-items:center;gap:8px}
#tf-flow .tf-legend i{display:inline-block;width:22px;border-top:1px solid #879aa5}#tf-flow .tf-legend .tf-dashed{border-top-style:dashed;margin-left:10px}
#tf-flow .tf-workspace{display:grid;grid-template-columns:minmax(0,1fr) 330px;border-top:1px solid var(--tf-line);border-bottom:1px solid var(--tf-line)}
#tf-flow .tf-map-scroll{overflow:auto;max-height:840px;min-width:0;background:#fafcfb}
#tf-flow .tf-map{position:relative;min-width:880px;padding:22px 18px;isolation:isolate}
#tf-flow .tf-edges{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:-1;overflow:visible}
#tf-flow .tf-lanes{display:flex;flex-direction:column;gap:28px}
#tf-flow .tf-lane{display:grid;grid-template-columns:110px minmax(0,1fr);gap:15px;align-items:center}
#tf-flow .tf-lane-label{font-size:13px;font-weight:600}#tf-flow .tf-lane-label small{display:block;font-size:11px;color:var(--tf-muted);font-weight:400;margin-top:4px}
#tf-flow .tf-chain{display:flex;align-items:stretch;gap:24px;min-width:0}
#tf-flow .tf-node{position:relative;display:flex;flex:1 0 124px;min-width:0;max-width:205px;flex-direction:column;align-items:flex-start;text-align:left;gap:6px;border:1px solid var(--tf-line);background:white;border-radius:7px;padding:11px 12px;font:inherit;color:inherit;cursor:pointer;box-shadow:0 2px 4px #20303904}
#tf-flow .tf-node:hover{border-color:#769a8e}#tf-flow .tf-node:focus-visible,#tf-flow .tf-ref:focus-visible{outline:3px solid #9dbdb2;outline-offset:2px}
#tf-flow .tf-node[aria-pressed=true]{border-color:var(--tf-green);box-shadow:0 0 0 1px var(--tf-green);background:var(--tf-wash)}
#tf-flow .tf-node.tf-connected{border-color:#93b5a9}#tf-flow .tf-node.tf-failed{border-left:3px solid #b04435}
#tf-flow .tf-kind{font-size:10px;letter-spacing:.04em;color:var(--tf-muted)}#tf-flow .tf-node-title{font-size:13px;font-weight:600;line-height:1.5;overflow-wrap:anywhere}
#tf-flow .tf-node-summary{font-size:11px;color:var(--tf-muted);line-height:1.55;overflow-wrap:anywhere}
#tf-flow .tf-node-title{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
#tf-flow .tf-metric{font-size:14px;font-weight:600;font-variant-numeric:tabular-nums;white-space:nowrap}
#tf-flow .tf-node-id{font-size:10px;color:var(--tf-muted);overflow-wrap:anywhere}#tf-flow .tf-empty{font-size:12px;color:var(--tf-muted);padding:14px 0}
#tf-flow .tf-main .tf-node{max-width:360px;flex-basis:240px}#tf-flow .tf-external{padding-top:18px;border-top:1px dashed #bdc9ce}
#tf-flow .tf-external .tf-node{border-style:dashed;max-width:330px;flex-basis:240px}
#tf-flow .tf-detail{padding:21px 23px;border-left:1px solid var(--tf-line);min-width:0;max-height:840px;overflow:auto;background:white}
#tf-flow .tf-detail h3{font-size:18px;line-height:1.5;margin:6px 0 12px;overflow-wrap:anywhere}#tf-flow .tf-detail h4{font-size:12px;color:var(--tf-muted);margin:20px 0 7px;font-weight:500}
#tf-flow .tf-detail p{font-size:13px;line-height:1.75;margin:5px 0;overflow-wrap:anywhere;white-space:pre-wrap}
#tf-flow .tf-detail code{font-size:11px;overflow-wrap:anywhere}#tf-flow .tf-detail table{font-size:12px}
#tf-flow .tf-detail th,#tf-flow .tf-detail td{border:0;border-bottom:1px solid #e8edef;padding:6px 2px;background:transparent}
#tf-flow .tf-detail td{text-align:right;font-variant-numeric:tabular-nums}#tf-flow .tf-ref{display:block;width:100%;background:transparent;border:0;border-bottom:1px solid #e8edef;text-align:left;padding:9px 0;cursor:pointer;color:var(--tf-green);font:inherit;font-size:12px;overflow-wrap:anywhere}
#tf-flow details{margin-top:15px}#tf-flow summary{cursor:pointer;font-size:12px;color:var(--tf-muted)}#tf-flow pre{font-size:11px;white-space:pre-wrap;overflow-wrap:anywhere;margin:8px 0}
#tf-flow .tf-empty-map{padding:24px;color:var(--tf-muted)}
@media(min-width:1550px){#tf-flow .tf-map{min-width:940px}#tf-flow .tf-node{flex-basis:135px}}
@media(max-width:1170px){#tf-flow .tf-workspace{grid-template-columns:minmax(0,1fr)}#tf-flow .tf-detail{border-left:0;border-top:1px solid var(--tf-line);max-height:480px}#tf-flow .tf-map-scroll{max-height:760px}}
@media(max-width:600px){#tf-flow .tf-heading{display:block}#tf-flow .tf-summary{display:block;margin-bottom:9px}#tf-flow h2{font-size:21px}#tf-flow .tf-legend{font-size:11px;flex-wrap:wrap}#tf-flow .tf-map{padding:20px 12px}#tf-flow .tf-detail{padding:18px 12px}}
@media print{#tf-flow .tf-workspace{display:block}#tf-flow .tf-map-scroll{max-height:none;overflow:visible}#tf-flow .tf-map{min-width:0}#tf-flow .tf-chain{flex-wrap:wrap}#tf-flow .tf-detail{display:none}#tf-flow .tf-toolbar{display:none}#tf-flow .tf-edges{display:none}#tf-flow .tf-lane{break-inside:avoid}}
"""

_JS = r"""
const root = document.getElementById('tf-flow');
const flow = JSON.parse(document.getElementById('tf-flow-data').textContent);
const nodes = new Map(flow.nodes.map(n => [n.id,n]));
const kinds = {track:'研究实例',message:'任务消息',source:'来源',idea:'想法',proposal:'实验提案',job:'实验请求',finding:'发现',stop:'停止研究',decision:'路线决定',report:'研究报告',final_evaluation:'外部留出评价',missing:'缺失引用',external_reference:'外部引用'};
const origins = {conjecture:'自主假说',history:'历史结果',literature:'文献启发',mixed:'多来源'};
const purposes = {explore:'探索',refine:'修订',replicate:'复现检验'};
const stages = {independent_proposals:'独立提案冻结',independent_experiments:'独立实验',sharing:'证据共享'};
const statuses = {completed:'已完成',succeeded:'已完成',failed:'失败',cancelled:'已取消',running:'执行中',reserved:'已预留',pending:'待执行',waiting:'等待',evaluated:'已评价',missing:'记录缺失',committed:'已冻结'};
const lanes = root.querySelector('.tf-lanes'), detail = root.querySelector('.tf-detail');
const svg = root.querySelector('.tf-edges'), map = root.querySelector('.tf-map');
const select = root.querySelector('.tf-track');
const recordSelect = root.querySelector('.tf-record');
let selected = null, visible = new Map();
const txt = value => value == null || value === '' ? '未记录' : typeof value === 'object' ? JSON.stringify(value,null,2) : String(value);
const el = (tag,cls,text) => {const e=document.createElement(tag);if(cls)e.className=cls;if(text!=null)e.textContent=String(text);return e;};
const pct = x => Number.isFinite(x) ? (x*100).toFixed(4)+'%' : '未记录';
const record = n => n.detail || {};
const metrics = n => n.metrics || {};
function modelLabel(n){const d=record(n),m=d.result?.model?.spec||d.result?.model||d.model||d.request?.model;return m ? [m.estimator || m.name || m.category,m.hyperparameters?.alpha!=null?'α='+m.hyperparameters.alpha:''].filter(Boolean).join(' · ') : '';}
function isMain(n){return (flow.lanes||[]).some(l=>l.track_id===n.track_id&&l.role==='main')||record(n).role==='main'||(record(nodes.get('track:'+n.track_id)||{}).context_only&&n.track_id==='main');}
function latestReports(items){let reports=items.filter(n=>n.kind==='report');const finals=reports.filter(n=>record(n).report_stage==='final');if(finals.length)reports=finals;return reports.length ? [reports.reduce((a,b)=>(record(a).created_at||0)>(record(b).created_at||0)?a:b)] : [];}
function parentChanges(n){
  const d=record(n),idea=nodes.get(d.idea_id),proposal=nodes.get(d.proposal_id),parents=[...new Set([...(record(idea||{}).parent_job_ids||[]),...(record(proposal||{}).parent_job_ids||[])])],changes=[];
  for(const id of parents){const p=nodes.get(id);if(!p)continue;const pd=record(p),fp=d.result?.protocol_fingerprint,pfp=pd.result?.protocol_fingerprint,old=metrics(p).CVRMSE,now=metrics(n).CVRMSE;
    if(d.comparable&&pd.comparable&&fp&&fp===pfp&&Number.isFinite(old)&&old>0&&Number.isFinite(now))changes.push({id,label:pd.result?.experiment_id||id,percent:((now-old)/old*100).toFixed(2)});
  }return changes;
}
function card(n){
  const d=record(n), b=el('button','tf-node');b.type='button';b.dataset.nodeId=n.id;b.setAttribute('aria-pressed',String(selected===n.id));
  b.append(el('span','tf-kind',kinds[n.kind]||n.kind));
  let title=n.label;
  if(n.kind==='idea')title=d.statement||n.label;
  if(n.kind==='job')title=modelLabel(n)||n.label;
  if(n.kind==='report')title=d.report_stage==='stage'?'阶段报告':isMain(n)?'主智能体综合报告':'独立研究报告';
  if(n.kind==='message')title=d.initial_task?'下发研究任务':'协调消息';
  if(n.kind==='track')title=n.label;
  if(n.kind==='final_evaluation')title='冻结方案 · 最终留出';
  b.append(el('span','tf-node-title',String(title||n.id).length>39?String(title).slice(0,38)+'…':title));
  if(n.kind==='job'){
    b.firstChild.textContent=(d.experiment_id||d.result?.experiment_id||'实验')+' · '+(statuses[n.status]||n.status||'未记录');
    const m=metrics(n);b.append(el('span','tf-metric',Number.isFinite(m.CVRMSE)?'CVRMSE '+pct(m.CVRMSE):'无可用验证指标'));
    const changes=parentChanges(n);
    b.append(el('span','tf-node-summary',Number.isFinite(m.CVRMSE)?'validate'+(changes.length===1?' · 较父实验 '+changes[0].percent+'%':''):statuses[n.status]||n.status||'未记录'));
    if(d.execution_label&&d.execution_label!=='执行方式未记录')b.append(el('span','tf-node-summary',d.execution_label));
  }else if(n.kind==='idea')b.append(el('span','tf-node-summary',origins[d.origin]||d.origin||'来源未记录'));
  else if(n.kind==='proposal')b.append(el('span','tf-node-summary',[purposes[d.purpose]||d.purpose,modelLabel(n)].filter(Boolean).join(' · ')));
  else if(n.kind==='stop')b.append(el('span','tf-node-summary','点击查看停止依据 · 不等于研究成功'));
  else if(n.kind==='final_evaluation'){
    const surfaces=d.surfaces||{};
    for(const [name,s] of Object.entries(surfaces))if(Number.isFinite(s.metrics?.CVRMSE))b.append(el('span','tf-metric',name+' · CVRMSE '+pct(s.metrics.CVRMSE)));
    b.append(el('span','tf-node-summary','仅供外部查看 · 不反馈给候选'));
  }else b.append(el('span','tf-node-summary',n.kind==='report'?'点击查看依据、结论与局限':n.kind==='message'?`${d.from||'未记录'} → ${d.to||'未记录'}`:statuses[n.status]||n.status||'已登记'));
  if(['failed','cancelled','interrupted','timeout'].includes(n.status))b.classList.add('tf-failed');
  b.addEventListener('click',()=>choose(n.id,true));visible.set(n.id,b);return b;
}
function lane(label,subtitle,items,cls='',trackNode=null){
  const row=el('div','tf-lane '+cls),name=el('div','tf-lane-label',label);name.append(el('small','',subtitle));
  if(trackNode){name.tabIndex=0;name.setAttribute('role','button');name.setAttribute('aria-label',label+'实例详情');name.addEventListener('click',()=>choose(trackNode.id));name.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose(trackNode.id);}});visible.set(trackNode.id,name);}
  row.append(name);const chain=el('div','tf-chain');items.forEach(n=>chain.append(card(n)));if(!items.length)chain.append(el('span','tf-empty','尚无已登记想法或实验'));row.append(chain);lanes.append(row);
}
function render(){
  visible=new Map();lanes.replaceChildren();const filter=select.value;
  const all=flow.nodes,allowed=n=>filter==='all'||n.track_id===filter;
  const relevantMessage=n=>n.kind==='message'&&(filter==='all'||record(n).from===filter||[filter,'all'].includes(record(n).to));
  const mainInputs=all.filter(n=>relevantMessage(n)&&isMain(n));
  const mainRecords=all.filter(n=>isMain(n)&&allowed(n));
  const mainTrack=all.find(n=>n.kind==='track'&&isMain(n));
  if(mainInputs.length)lane('主智能体','任务与协调',mainInputs,'tf-main',mainTrack);
  for(const l of flow.lanes||[]){
    if(l.role==='main'||(filter!=='all'&&l.track_id!==filter))continue;
    const owned=all.filter(n=>n.track_id===l.track_id),chosenReports=latestReports(owned);
    const backbone=owned.filter(n=>['idea','proposal','job','stop','message'].includes(n.kind)).sort((a,b)=>(record(a).created_at||0)-(record(b).created_at||0));
    lane(l.label||l.track_id,statuses[l.status]||l.status||'独立研究轨迹',[...backbone,...chosenReports],'',owned.find(n=>n.kind==='track'));
  }
  const mainWork=mainRecords.filter(n=>['idea','proposal','job','stop','decision'].includes(n.kind));
  const summaries=latestReports(mainRecords);
  if(mainWork.length||summaries.length)lane('主智能体','比较与综合',[...mainWork,...summaries],'tf-main',mainInputs.length?null:mainTrack);
  const external=all.filter(n=>n.kind==='final_evaluation'&&(filter==='all'||record(n).track_id_selected===filter));
  if(external.length)lane('外部评价','研究结束后查看',external,'tf-external');
  recordSelect.replaceChildren(el('option','','选择来源、发现或历史报告'));recordSelect.firstChild.value='';
  for(const n of all.filter(n=>allowed(n)||relevantMessage(n))){const o=el('option','',`${n.track_id||'外部'} · ${kinds[n.kind]||n.kind} · ${String(n.label).slice(0,44)}`);o.value=n.id;recordSelect.append(o);}
  if(!lanes.children.length)lanes.append(el('p','tf-empty-map','当前范围尚无可绘制的研究记录。'));
  requestAnimationFrame(drawEdges);
}
function drawEdges(){
  svg.replaceChildren();const NS='http://www.w3.org/2000/svg',make=(tag,attrs)=>{const x=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))x.setAttribute(k,v);return x;};
  svg.setAttribute('width',map.scrollWidth);svg.setAttribute('height',map.scrollHeight);
  const defs=make('defs',{}),mark=make('marker',{id:'tf-arrow',viewBox:'0 0 8 8',refX:7,refY:4,markerWidth:5,markerHeight:5,orient:'auto-start-reverse'});mark.append(make('path',{d:'M 0 0 L 8 4 L 0 8 z',fill:'#839a92'}));defs.append(mark);svg.append(defs);
  const bounds=map.getBoundingClientRect();
  for(const edge of flow.edges){
    const from=visible.get(edge.from),to=visible.get(edge.to);if(!from||!to||from===to)continue;
    const source=nodes.get(edge.from),target=nodes.get(edge.to);
    // The full evidence graph remains in the inspector. Avoid drawing every job
    // directly into a team report when its own report already forms the overview.
    if(source.track_id!==target.track_id&&isMain(target)&&['report','decision'].includes(target.kind)&&source.kind!=='report')continue;
    const a=from.getBoundingClientRect(),b=to.getBoundingClientRect();
    const same=Math.abs(a.top-b.top)<35;
    let x1=(same?a.right:a.left+a.width/2)-bounds.left,y1=(same?a.top+a.height/2:a.bottom)-bounds.top;
    let x2=(same?b.left:b.left+b.width/2)-bounds.left,y2=(same?b.top+b.height/2:b.top)-bounds.top;
    if(!same&&b.top<a.top){y1=a.top-bounds.top;y2=b.bottom-bounds.top;}
    let d=same?`M ${x1} ${y1} C ${x1+12} ${y1}, ${x2-12} ${y2}, ${x2} ${y2}`:`M ${x1} ${y1} C ${x1} ${(y1+y2)/2}, ${x2} ${(y1+y2)/2}, ${x2} ${y2}`;
    if(!same&&((source.kind==='report'&&target.kind==='report')||target.kind==='final_evaluation')){
      const gutter=Math.max(a.right,b.right)-bounds.left+8;
      d=`M ${a.right-bounds.left} ${a.top+a.height/2-bounds.top} H ${gutter} V ${y2-10} H ${x2} V ${y2}`;
    }
    const active=edge.from===selected||edge.to===selected;
    const path=make('path',{d,fill:'none',stroke:active?'#176553':'#aabbb5','stroke-width':active?1.8:1,'marker-end':'url(#tf-arrow)',opacity:selected&&!active ? .35 : .8});
    if(['membership','owns','author','recipient','sender','dispatch','message_recipient','message_sender','message_sent','message_to','track_record','duplicate_configuration'].includes(edge.relation))path.setAttribute('stroke-dasharray','4 4');
    svg.append(path);
  }
}
function field(label,value){if(value==null||value==='')return;detail.append(el('h4','',label),el('p','',txt(value)));}
function metricTable(m,label){if(!Object.keys(m).length)return;detail.append(el('h4','',label));const table=el('table'),tbody=el('tbody');for(const [key,value] of Object.entries(m)){if(!Number.isFinite(value))continue;const row=el('tr');row.append(el('th','',key),el('td','',['CVRMSE','MAPE','NMBE'].includes(key)?pct(value):Number(value).toFixed(6)));tbody.append(row);}table.append(tbody);detail.append(table);}
function choose(id,userAction=false){
  const n=nodes.get(id);if(!n)return;selected=id;
  recordSelect.value=[...recordSelect.options].some(o=>o.value===id)?id:'';
  for(const [key,b] of visible){if(b.tagName==='BUTTON')b.setAttribute('aria-pressed',String(key===id));b.classList.toggle('tf-connected',key!==id&&flow.edges.some(e=>(e.from===id&&e.to===key)||(e.to===id&&e.from===key)));}
  detail.replaceChildren();const d=record(n);
  detail.append(el('span','tf-eyebrow',kinds[n.kind]||n.kind),el('h3','',n.label||id),el('code','',id));
  field('所属轨迹',n.track_id);if(n.status)field('状态',statuses[n.status]||n.status);
  field('想法 / 观察',d.statement);if(d.origin)field('想法来源',origins[d.origin]||d.origin);
  field('为什么从这里着手',d.reason);field('实验前预测',d.prediction);field('怎样证伪',d.falsification);
  if(n.kind==='proposal'){
    field('冻结版本',d.version);field('研究用途',purposes[d.purpose]||d.purpose);
    field('冻结模型方案',d.model);field('预期成本',d.expected_cost);
    field('依据的历史实验',d.parent_job_ids);field('实验配置指纹',d.experiment_fingerprint);
    if(d.version===1&&d.status==='committed')field('提案阶段','首轮独立冻结；后续修订保留为新版本。');
  }
  if(n.kind==='stop'){field('停止依据',d.evidence_ids);field('已覆盖实验',d.job_ids);field('决定边界','停止是研究过程决定，不自动表示目标已达到或假说获得验证。');}
  if(n.kind==='job'){
    field('执行方式',d.execution_label);field('冻结提案',d.proposal_id);
    field('复用来源（非独立复现）',d.reused_from_job_id);field('相同配置的既有请求（非启发关系）',d.duplicate_of_job_id);
    field('实验配置指纹',d.experiment_fingerprint);
    field(d.result?.model?'结果记录的模型':'请求模型（结果未登记实际模型）',modelLabel(n)||d.model||d.request?.model);metricTable(metrics(n),d.reused_from_job_id?'复用的验证指标 · validate':'验证集实测 · validate');
    if(!Object.keys(metrics(n)).length)field('指标状态','未获得可用于展示的验证指标；缺失或失败不按零分计算。');
    field('错误',d.error||d.result?.error);
    if(d.comparable===false)field('比较口径','此实验不满足当前冻结协议的可比条件；只保留其已登记事实。');
    for(const change of parentChanges(n))field('相对已登记父实验的 CVRMSE 变化',`${change.label} → 当前：${change.percent}%（负数表示误差降低；描述性比较）`);
  }
  field('任务内容',d.text);field('摘要',d.summary);field('报告正文',d.body);field('结果解释',d.interpretation);field('适用条件与局限',d.limitations);
  field('登记时的研究阶段',stages[d.research_stage]||d.research_stage);
  field('报告阶段',d.report_stage);
  if(n.kind==='source'){field('来源类型',d.kind);field('实际阅读范围',d.read_scope);field('登记核验',d.verification);field('URL',d.url);field('DOI',d.doi);}
  if(n.kind==='final_evaluation'){
    field('用途','仅供外部查看，不反馈给内部研究智能体。');field('选择依据',d.selection_reason);
    for(const [name,s] of Object.entries(d.surfaces||{}))metricTable(s.metrics||{},'留出 '+name);
    if(!Object.keys(d.surfaces||{}).length)field('留出状态','没有可用留出指标，不能判断最终泛化结果。');
  }
  const incoming=flow.edges.filter(e=>e.to===id),outgoing=flow.edges.filter(e=>e.from===id);
  for(const [heading,list,direction] of [['它依据什么',incoming,'from'],['哪些记录引用或关联它',outgoing,'to']]){
    if(!list.length)continue;detail.append(el('h4','',heading));
    for(const edge of list){const target=nodes.get(edge[direction]);if(!target)continue;const button=el('button','tf-ref',(edge.label||edge.relation)+' · '+target.label);button.type='button';button.addEventListener('click',()=>choose(target.id));detail.append(button);}
  }
  const disclosure=el('details'),summary=el('summary','','完整登记记录');disclosure.append(summary,el('pre','',JSON.stringify(d,null,2)));detail.append(disclosure);
  if(innerWidth<=1170){const back=el('button','tf-ref','返回流程图');back.type='button';back.addEventListener('click',()=>root.querySelector('.tf-map-scroll').scrollIntoView({block:'start'}));detail.prepend(back);}
  detail.scrollTop=0;
  if(userAction&&innerWidth<=1170)detail.scrollIntoView({block:'start'});
  requestAnimationFrame(drawEdges);
}
for(const laneData of flow.lanes||[]){const option=el('option','',laneData.label||laneData.track_id);option.value=laneData.track_id;select.append(option);}
const jobs=flow.nodes.filter(n=>n.kind==='job'),reused=jobs.filter(n=>record(n).reused_from_job_id).length;
root.querySelector('.tf-summary').textContent=`${(flow.lanes||[]).length} 个实例 · ${flow.nodes.filter(n=>n.kind==='idea').length} 个想法 · ${jobs.length} 个实验请求`+(reused?` · ${reused} 个复用`:'')+(flow.research_stage?' · '+(stages[flow.research_stage]||flow.research_stage):'');
select.addEventListener('change',()=>{selected=null;render();const first=[...visible.keys()].find(id=>nodes.get(id)?.kind==='job')||[...visible.keys()][0];if(first)choose(first);else detail.replaceChildren(el('p','','当前范围尚无研究记录。'));});
recordSelect.addEventListener('change',()=>{if(recordSelect.value)choose(recordSelect.value,true);});
render();const final=flow.nodes.find(n=>n.kind==='final_evaluation');const initial=record(final||{}).job_id||flow.nodes.find(n=>n.kind==='job')?.id||flow.nodes[0]?.id;if(initial&&nodes.has(initial))choose(initial);
new ResizeObserver(()=>requestAnimationFrame(drawEdges)).observe(map);
"""
