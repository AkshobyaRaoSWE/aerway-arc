"""Synthetic reference tests, never an instrument or clinical validation."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from arc.reference import labels_from_pressure, protocol_digest, validate_awake_reference, validate_protocol


@pytest.fixture
def protocol():
    return {"schema_version": 1, "status": "synthetic", "protocol_id": "test-pressure-rule", "protocol_version": "1",
            "instrument_id": "synthetic-only", "instrument_model": "generated pressure samples",
            "placement": "synthetic tongue-palate reference, not body measurement", "calibration_id": "test-calibration",
            "calibration_record": "generator identity scaling", "source_record": "generated test waveform",
            "approved_by": "synthetic test fixture, no clinical approval", "approval_reference": "test-only",
            "stream": "reference_pressure", "pressure_threshold_kpa": 5., "hold_min_s": .2, "interruption_min_s": .3}


def convert(values, protocol, valid=None, rate=10, start=0.):
    values = np.asarray(values, dtype=float)[:, None]
    if valid is None:
        valid = np.ones(len(values), dtype=bool)
    return labels_from_pressure(values, np.asarray(valid, dtype=bool), rate, start, protocol)


def simple_session(values, protocol):
    samples = np.asarray(values, dtype=float)[:, None]
    valid = np.ones(len(samples), dtype=bool)
    return SimpleNamespace(
        metadata={"synthetic": True, "awake_reference": deepcopy(protocol),
                  "acquisition_config": {"reference_alignment": {"method": "shared synthetic clock", "max_error_s": 0.}}},
        streams={"reference_pressure": {"sample_rate_hz": 10, "start_time_s": 0., "n_samples": len(samples),
                                          "units": ["kPa"], "channels": ["tongue_palate_pressure"]}},
        labels=labels_from_pressure(samples, valid, 10, 0., protocol),
        read=lambda name, start, end: (samples, valid),
    )


def test_threshold_and_duration_equalities(protocol):
    labels = convert([5, 5, 4, 4, 4], protocol, start=2.)
    assert [(x["state"], x["start_s"], x["end_s"]) for x in labels] == [("positive", 2., 2.2), ("negative", 2.2, 2.5)]
    assert all(x["reference_protocol_digest"] == protocol_digest(protocol) for x in labels)


def test_short_holds_and_dips_excluded_not_relabelled(protocol):
    labels = convert([6, 4, 4, 6, 6], protocol)
    assert [(x["state"], x["start_s"], x["end_s"]) for x in labels] == [("exclude", 0., .3), ("positive", .3, .5)]


def test_invalid_reference_does_not_bridge_interruption(protocol):
    labels = convert([1] * 5, protocol, [True, True, False, True, True])
    assert len(labels) == 1 and labels[0]["state"] == "exclude"
    assert labels[0]["end_s"] == .5


def test_invalid_reference_is_never_positive(protocol):
    labels = convert([6, 6, np.nan, 6, 6], protocol, [True, True, False, True, True])
    assert [label["state"] for label in labels] == ["positive", "exclude", "positive"]
    with pytest.raises(ValueError, match="finite"):
        convert([6, np.nan], protocol)


@pytest.mark.parametrize("field", ["pressure_threshold_kpa", "hold_min_s", "interruption_min_s"])
@pytest.mark.parametrize("value", [None, True, -1., 0., float("nan")])
def test_no_implicit_or_nonfinite_clinical_defaults(protocol, field, value):
    protocol[field] = value
    with pytest.raises(ValueError):
        convert([6, 6, 1, 1, 1], protocol)


@pytest.mark.parametrize("field", ["instrument_id", "instrument_model", "placement", "calibration_record", "approved_by", "approval_reference"])
def test_missing_provenance_rejected(protocol, field):
    del protocol[field]
    with pytest.raises(ValueError, match=field):
        validate_protocol(protocol)


def test_unapproved_reference_and_synthetic_approval_cannot_pass_real_gate(protocol):
    session = simple_session([6, 6, 1, 1, 1], protocol)
    session.metadata["synthetic"] = False
    with pytest.raises(ValueError, match="status"):
        validate_awake_reference(session)
    session.metadata["awake_reference"]["status"] = "pending"
    with pytest.raises(ValueError, match="unapproved"):
        validate_awake_reference(session)


def test_metadata_gate_never_reads_held_out_waveform(protocol):
    session = simple_session([6, 6, 1, 1, 1], protocol)
    def forbidden_read(*args):
        raise AssertionError("held-out pressure was read")
    session.read = forbidden_read
    assert validate_awake_reference(session) == protocol_digest(protocol)


def test_exact_reproduction_accepts_explicit_exclusion_overlay(protocol):
    session = simple_session([6, 6, 1, 1, 1], protocol)
    session.labels.append({"task": "awake", "state": "exclude", "start_s": .25, "end_s": .35, "provenance": "reviewer exclusion"})
    assert validate_awake_reference(session, verify_labels=True) == protocol_digest(protocol)


def test_changed_label_or_protocol_rejected(protocol):
    session = simple_session([6, 6, 1, 1, 1], protocol)
    session.labels[0]["state"] = "negative"
    with pytest.raises(ValueError, match="reproduce"):
        validate_awake_reference(session, verify_labels=True)
    session = simple_session([6, 6, 1, 1, 1], protocol)
    session.metadata["awake_reference"]["pressure_threshold_kpa"] = 7.
    with pytest.raises(ValueError, match="digest"):
        validate_awake_reference(session, verify_labels=True)


def test_missing_labels_rejected_even_with_pressure(protocol):
    session = simple_session([6, 6, 1, 1, 1], protocol)
    session.labels = []
    with pytest.raises(ValueError, match="reproduce"):
        validate_awake_reference(session, verify_labels=True)


def test_emg_cannot_be_named_as_reference(protocol):
    protocol["stream"] = "emg"
    with pytest.raises(ValueError, match="separate"):
        validate_protocol(protocol)


def test_incompatible_reference_units_rejected(protocol):
    session = simple_session([6, 6, 1, 1, 1], protocol)
    session.streams["reference_pressure"]["units"] = ["uV"]
    with pytest.raises(ValueError, match="kPa"):
        validate_awake_reference(session)


def test_scored_labels_outside_reference_are_rejected(protocol):
    session = simple_session([6, 6, 1, 1, 1], protocol)
    session.labels[0]["end_s"] = 2.
    with pytest.raises(ValueError, match="outside"):
        validate_awake_reference(session, verify_labels=True)


def test_pressure_labels_use_plain_json_number_types(protocol):
    labels = convert([5, 5, 4, 4, 4], protocol)
    assert all(type(label["start_s"]) is float and type(label["end_s"]) is float for label in labels)
