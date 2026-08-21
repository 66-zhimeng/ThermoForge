"""逐台冷却塔风机建模表：频率 → 功耗。

产出 `WX_CT_FAN_UNIT`，对象 `CTF0101..CTF0403`（4 塔 × 3 台），物模型
`ct_fan_unit.v1`。

**为什么值得单独做**：塔风机总功耗（均值 217 kW）是冷站里唯一还没建模的
能耗项，而「调频省风机电 vs 出水温升高多耗冷机电」这道题需要它才能闭合。
逐台功耗与频率都是实测量，关系干净（corr 0.973~0.992）。

**两级结构**（实测支持，见 `ct_fan_unit.v1` 注释）：塔内三台高度同步
（极差中位 0.16~0.45 Hz），塔间差异明显（p95 7.59 Hz）。所以单台是物理
单元、**塔是控制单元**，字段同时给到两级。

**做不了什么**：塔侧没有任何风量测点（`ct_0i.instant_flow` 整表为空，
TFDC-306），所以只能到功耗层，无法建「频率 → 风量」。

用法::

    python scripts/build_ct_fan_dataset.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from thermoforge_core.contracts.tfdc import (
    ObjectRecord,
    TfdcDataset,
    TfdcManifest,
    VariableRecord,
)
from thermoforge_data.importer import default_registry, import_parsed
from thermoforge_research.tools import ToolContext

TOWERS = range(1, 5)
FANS = range(1, 4)
MODEL_ID = "ct_fan_unit.v1"

COLS: list[tuple[str, str, str, str, str]] = [
    ("power", "kW", "float", "target", "measured"),
    ("frequency", "Hz", "float", "control", "measured"),
    ("tower_fan_count", "1", "integer", "state", "measured"),
    ("tower_freq_mean", "Hz", "float", "control", "measured"),
    ("bank_fan_count", "1", "integer", "state", "measured"),
    ("bank_freq_mean", "Hz", "float", "state", "measured"),
    ("ambient_t", "Cel", "float", "disturbance", "measured"),
    ("ambient_h", "%", "float", "disturbance", "measured"),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", default="WX_2025_HVAC@rev_0001")
    ap.add_argument("--vault-root", type=Path, default=Path("vault"))
    ap.add_argument("--research-root", type=Path, default=Path("research"))
    ap.add_argument("--models-root", type=Path, default=Path("models"))
    args = ap.parse_args(argv)

    ctx = ToolContext(vault_root=args.vault_root, research_root=args.research_root,
                      models_root=args.models_root, actor="human")
    df = ctx.vault.load_data(args.ref).to_pandas()
    ts = pd.to_datetime(df["timestamp"])
    num = lambda c: pd.to_numeric(df[c], errors="coerce")  # noqa: E731

    registry = default_registry()
    if registry.get(MODEL_ID) is None:
        raise SystemExit(f"物模型未注册: {MODEL_ID}")

    on = {(i, j): (df[f"ct_0{i}_f_0{j}.status_run"] == True).fillna(False)  # noqa: E712
          for i in TOWERS for j in FANS}
    freq = {k: num(f"ct_0{k[0]}_f_0{k[1]}.status_frequency") for k in on}
    power = {k: num(f"ct_0{k[0]}_f_0{k[1]}.power") for k in on}

    bank_count = sum(v.astype(int) for v in on.values())
    bank_fsum = sum(freq[k].where(on[k], 0.0).fillna(0.0) for k in on)
    bank_fmean = bank_fsum / bank_count.replace(0, np.nan)
    tower_count = {i: sum(on[(i, j)].astype(int) for j in FANS) for i in TOWERS}
    tower_fmean = {
        i: (sum(freq[(i, j)].where(on[(i, j)], 0.0).fillna(0.0) for j in FANS)
            / tower_count[i].replace(0, np.nan))
        for i in TOWERS
    }
    amb_t = num("environment_parameters.ambient_t")
    amb_h = num("environment_parameters.ambient_h")

    objects: list[str] = []
    values: dict[str, dict[str, pd.Series]] = {}
    keep_any = pd.Series(False, index=df.index)
    for i in TOWERS:
        for j in FANS:
            obj = f"CTF0{i}0{j}"
            objects.append(obj)
            k = (i, j)
            run = (on[k] & power[k].notna() & (power[k] > 0)
                   & freq[k].notna() & (freq[k] > 5))
            values[obj] = {
                "power": power[k].where(run),
                "frequency": freq[k].where(run),
                "tower_fan_count": tower_count[i].where(run),
                "tower_freq_mean": tower_fmean[i].where(run),
                "bank_fan_count": bank_count.where(run),
                "bank_freq_mean": bank_fmean.where(run),
                "ambient_t": amb_t.where(run),
                "ambient_h": amb_h.where(run),
            }
            keep_any |= run.fillna(False)

    axis = sorted(set(ts[keep_any]))
    index = {t: k for k, t in enumerate(axis)}
    print(f"逐台塔风机表时间轴: {len(axis)} 个时刻 × {len(objects)} 台")

    columns: dict[str, list] = {}
    for obj in objects:
        for code, _u, dtype, _r, _sk in COLS:
            col: list = [None] * len(axis)
            for t, v in zip(ts, values[obj][code]):
                if pd.notna(v) and t in index:
                    col[index[t]] = int(v) if dtype == "integer" else float(v)
            columns[f"{obj}.{code}"] = col

    ds = TfdcDataset(
        manifest=TfdcManifest(
            contract="TFDC", contract_version="1.0",
            dataset_id="WX_CT_FAN_UNIT", dataset_version=1, site_id="WX",
            timezone="Asia/Shanghai", time_resolution="900s",
            source_system="derived",
            description="逐台冷却塔风机建模表（4 塔 × 3 台）：频率 → 功耗。"
                        "塔内三台同调、塔间有差异，故同时给塔级与塔群级字段。"
                        "塔侧无风量测点（TFDC-306），只能到功耗层。",
        ),
        objects=[ObjectRecord(object_id=o, object_model_id=MODEL_ID,
                              object_name=o) for o in objects],
        variables=[
            VariableRecord(
                variable_id=f"{o}.{code}", object_id=o, property_code=code,
                unit=unit, dtype=dtype, role=role, source_kind=sk,
                nullable=True, sample_period="900s")
            for o in objects for code, unit, dtype, role, sk in COLS
        ],
    )
    res = import_parsed(ds, axis, columns, registry=registry)
    if not res.ok:
        for d in res.diagnostics[:10]:
            print("  ", d.code, d.level, getattr(d, "location", ""))
        raise SystemExit("逐台塔风机表导入失败")
    ref = ctx.vault.store(res, lineage={
        "derived_from": args.ref,
        "scope": "逐台冷却塔风机（频率→功耗）",
        "note": "塔内三台频率极差中位 0.16~0.45 Hz（同调），塔间 p95 7.59 Hz",
    })
    print(f"逐台塔风机表 {ref}  rows={res.table.num_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
