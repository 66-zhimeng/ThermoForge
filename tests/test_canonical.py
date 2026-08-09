"""Canonical JSON 与哈希稳定性测试（conventions.md §5.1）。"""

import json

from hypothesis import given, settings
from hypothesis import strategies as st

from thermoforge_core.canonical import (
    EXCLUDED_KEYS,
    canonical_hash,
    canonical_json,
    sha256_hex,
)

_json_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(),
)
json_values = st.recursive(
    _json_scalars,
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(st.text(min_size=1, max_size=8), children, max_size=5),
    ),
    max_leaves=20,
)


def test_known_vector():
    # 键按码点排序、无空格分隔符、排除字段删除、null 键删除、5.0 保留类型
    obj = {"b": 1, "a": 5.0, "name_zh": "名称", "c": None, "description": "x"}
    assert canonical_json(obj) == '{"a":5.0,"b":1}'


def test_non_ascii_not_escaped():
    assert canonical_json({"k": "冷水机"}) == '{"k":"冷水机"}'


def test_excluded_keys_complete():
    assert EXCLUDED_KEYS == frozenset(
        {"description", "name_zh", "name_en", "created_at", "author", "comment", "tags"}
    )


def test_arrays_keep_order_by_default():
    assert canonical_json({"a": [2, 1]}) == '{"a":[2,1]}'


def test_set_fields_sorted():
    a = {"objects": ["CH-02", "CH-01"], "metrics": ["RMSE", "MAE"]}
    b = {"objects": ["CH-01", "CH-02"], "metrics": ["MAE", "RMSE"]}
    assert canonical_json(a) != canonical_json(b)
    sf = {"objects", "metrics"}
    assert canonical_json(a, sf) == canonical_json(b, sf)
    assert canonical_hash(a, sf) == canonical_hash(b, sf)


def test_negative_zero_normalized():
    # §4.3：-0.0 规范化为 0.0
    assert canonical_json({"x": -0.0}) == '{"x":0.0}'


def test_null_equivalent_to_missing():
    assert canonical_json({"a": 1, "b": None}) == canonical_json({"a": 1})


@settings(max_examples=200)
@given(json_values)
def test_idempotent(value):
    # implementation-notes §13.3：canonical_json(parse(canonical_json(x))) == canonical_json(x)
    once = canonical_json(value, {"objects", "metrics", "features"})
    twice = canonical_json(json.loads(once), {"objects", "metrics", "features"})
    assert once == twice


@settings(max_examples=200)
@given(json_values)
def test_hash_stable(value):
    assert canonical_hash(value) == canonical_hash(json.loads(canonical_json(value)))
    assert len(canonical_hash(value)) == 64


def test_sha256_hex_accepts_str_and_bytes():
    assert sha256_hex("abc") == sha256_hex(b"abc")
