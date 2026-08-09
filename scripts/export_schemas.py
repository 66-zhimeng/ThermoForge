"""导出五份契约的 JSON Schema 到 contracts/ 目录。

用法：`.venv/Scripts/python scripts/export_schemas.py`
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoforge_core.contracts.export import export_schemas

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "contracts"
    for path in export_schemas(root):
        print(f"written: {path}")
