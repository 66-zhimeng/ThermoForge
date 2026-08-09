"""ThermoForge 研究闭环层（Phase 2：确定性研究内核）。

- `ledger`：Research Ledger（goals/hypotheses/experiments/findings/
  decisions/models，research-loop §8）。
- `metrics`：RMSE/MAE/MAPE/CVRMSE/NMBE 单一实现（implementation-notes §5）。
- `splits`：时间边界切分 + purge/embargo + 泄漏检测（§4）。
- `physics_checks`：可证伪硬约束与单调性受控扰动扫描（§6）。
- `runner`：Experiment Runner，子进程隔离 + 种子清单 + 环境指纹 +
  机器判定的复现性（§7）。
"""

from . import errors, ledger, metrics, physics_checks, runner, splits

__all__ = ["errors", "ledger", "metrics", "physics_checks", "runner", "splits"]
