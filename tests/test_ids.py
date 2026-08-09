"""顺序 ID 分配器测试（conventions.md §1.3）。"""

import json

import pytest

from thermoforge_core.ids import IdAllocationError, IdAllocator


def test_allocate_increments_with_zero_padding(tmp_path):
    alloc = IdAllocator(tmp_path)
    assert alloc.allocate("EXP-") == "EXP-0001"
    assert alloc.allocate("EXP-") == "EXP-0002"
    assert alloc.allocate("RG-") == "RG-0001"
    assert alloc.peek("EXP-") == 2
    assert alloc.peek("M-") == 0


def test_counter_persists_across_instances(tmp_path):
    assert IdAllocator(tmp_path).allocate("VIEW-") == "VIEW-0001"
    assert IdAllocator(tmp_path).allocate("VIEW-") == "VIEW-0002"


def test_extends_beyond_9999_without_reset(tmp_path):
    # 超过 9999 后自然扩展位数，不重置、不复用、不回收
    counter_file = tmp_path / "id_counters.json"
    tmp_path.mkdir(exist_ok=True)
    with open(counter_file, "w", encoding="utf-8", newline="\n") as fp:
        json.dump({"RG-": 9999}, fp)
    alloc = IdAllocator(tmp_path)
    assert alloc.allocate("RG-") == "RG-10000"
    assert alloc.allocate("RG-") == "RG-10001"


def test_counter_file_valid_and_no_temp_left(tmp_path):
    alloc = IdAllocator(tmp_path)
    for _ in range(5):
        alloc.allocate("H-")
    with open(tmp_path / "id_counters.json", encoding="utf-8") as fp:
        data = json.load(fp)
    assert data == {"H-": 5}
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".id_counters.")]
    assert leftovers == []


def test_unknown_prefix_rejected(tmp_path):
    with pytest.raises(IdAllocationError):
        IdAllocator(tmp_path).allocate("XX-")
    with pytest.raises(IdAllocationError):
        IdAllocator(tmp_path).peek("EXP")  # 缺少连字符
