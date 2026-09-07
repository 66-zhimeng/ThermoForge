"""Exact identity controls result reuse; similar research narratives do not."""

from copy import deepcopy
import re

import pytest

from thermoforge_v2.experiment_identity import experiment_fingerprint


def data_model(**params):
    return {"category": "data", "estimator": "ridge", "hyperparameters": params}


def lab_model(reference="custom@v1", **params):
    return {"category": "lab", "hyperparameters": {"lab": reference, **params}}


def test_canonical_model_defaults_and_key_order_have_one_identity_without_mutation():
    model = data_model(alpha=1.0, label="中文")
    before = deepcopy(model)
    reordered = {"hyperparameters": {"label": "中文", "alpha": 1.0},
                 "estimator": "ridge", "category": "data", "physics": None,
                 "residual": None}
    identity = experiment_fingerprint("frozen-protocol", model)
    assert re.fullmatch(r"[0-9a-f]{64}", identity)
    assert identity == experiment_fingerprint("frozen-protocol", reordered)
    assert model == before


def test_protocol_and_explicit_model_changes_do_not_reuse_identity():
    base = experiment_fingerprint("protocol-data-seed-split-env-v1", data_model(alpha=1.0))
    assert base != experiment_fingerprint("protocol-data-seed-split-env-v2", data_model(alpha=1.0))
    assert base != experiment_fingerprint("protocol-data-seed-split-env-v1", data_model(alpha=2.0))
    assert base != experiment_fingerprint("protocol-data-seed-split-env-v1",
                                         data_model(alpha=1.0) | {"estimator": "linear"})


def test_lab_source_hash_is_required_even_when_reference_has_version():
    with pytest.raises(ValueError, match="SHA-256"):
        experiment_fingerprint("protocol", lab_model())
    with pytest.raises(ValueError, match="SHA-256"):
        experiment_fingerprint("protocol", lab_model(), "not-a-content-hash")


def test_lab_source_reference_and_parameters_all_belong_to_exact_identity():
    source = "a" * 64
    base = experiment_fingerprint("protocol", lab_model(alpha=1), source)
    assert base == experiment_fingerprint("protocol", lab_model(alpha=1), source.upper())
    assert base != experiment_fingerprint("protocol", lab_model(alpha=1), "b" * 64)
    assert base != experiment_fingerprint("protocol", lab_model("renamed@v1", alpha=1), source)
    assert base != experiment_fingerprint("protocol", lab_model("custom@v2", alpha=1), source)
    assert base != experiment_fingerprint("protocol", lab_model(alpha=2), source)
    assert base != experiment_fingerprint("other-protocol", lab_model(alpha=1), source)


def test_json_parameter_types_are_not_assumed_equivalent_for_arbitrary_lab_code():
    identities = {experiment_fingerprint("protocol", lab_model(value=value), "a" * 64)
                  for value in [1, 1.0, "1", True]}
    assert len(identities) == 4


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_parameters_cannot_acquire_reusable_identity(value):
    with pytest.raises(ValueError, match="有限"):
        experiment_fingerprint("protocol", data_model(alpha=value))


@pytest.mark.parametrize("protocol", [None, "", "   ", 17])
def test_missing_protocol_cannot_acquire_identity(protocol):
    with pytest.raises(ValueError, match="冻结评价协议"):
        experiment_fingerprint(protocol, data_model())


def test_builtin_identity_rejects_misplaced_lab_hash_and_invalid_model():
    with pytest.raises(ValueError, match="内置模型"):
        experiment_fingerprint("protocol", data_model(), "a" * 64)
    with pytest.raises(ValueError):
        experiment_fingerprint("protocol", {"category": "data"})
    with pytest.raises(ValueError, match="模型规格对象"):
        experiment_fingerprint("protocol", "ridge")
