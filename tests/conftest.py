"""共享测试工具：仓库根目录与 fixture 路径。"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
DOCS_DIR = REPO_ROOT / "docs"
CONTRACTS_DIR = REPO_ROOT / "contracts"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR
