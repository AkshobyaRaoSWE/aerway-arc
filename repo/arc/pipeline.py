"""Frozen participant splits, validation-only selection, and held-out scoring."""
from __future__ import annotations

import json
import platform
from importlib.metadata import version
from pathlib import Path
import time

import numpy as np

from . import __version__
from .dataset import (DEFAULTS, dataset_identity, fingerprint, input_signature,
                      iter_windows, reference_events, validate_config, write_json)
from .evaluation import score_predictions, validate_splits
from .reference import validate_awake_reference
from .models import fit_awake_threshold, fit_baseline, predict_proba, validate_bundle


def collect(sessions, ids, task, config, signature):
    rows, features, names = [], [], None
    for session in sessions:
        if session.metadata["participant_id"] not in ids:
            continue
        for row, vector, current_names in iter_windows(session, task, config, signature):
            if vector is not None:
                if names is None:
                    names = current_names
                elif names != current_names:
                    raise ValueError("feature order changed across sessions")
                row["feature_index"] = len(features)
                features.append(vector)
            rows.append(row)
    return rows, np.asarray(features), names


def _labelled(rows, features):
    eligible = [r for r in rows if r["valid"] and r["label"] is not None]
    if not eligible or {r["label"] for r in eligible} != {0, 1}:
        raise ValueError("each train/validation partition needs valid independent labels for both classes")
    return features[[r["feature_index"] for r in eligible]], np.asarray([r["label"] for r in eligible])


def _assign(rows, probabilities):
    return [{**{key: value for key, value in row.items() if key != "feature_index"},
             "probability": float(probabilities[row["feature_index"]]) if row["valid"] else None}
            for row in rows]


def select_threshold(labels, probabilities):
    """Macro-average sensitivity/specificity; no test participant is consulted."""
    candidates = []
    for threshold in np.linspace(0.05, 0.95, 19):
        predicted = probabilities >= threshold
        sensitivity = np.mean(predicted[labels == 1])
        specificity = np.mean(~predicted[labels == 0])
        score = float((sensitivity + specificity) / 2)
        candidates.append((score, -abs(float(threshold) - 0.5), -float(threshold)))
    best = max(candidates)
    return -best[2], best[0]


def train(sessions, splits, task, output, config=None, *, seed=42, allow_synthetic=False,
          min_duration_s=0.0, gap_merge_s=0.0):
    config = dict(DEFAULTS[task] if config is None else config)
    validate_config(task, config)
    validate_splits(splits, [s.metadata["participant_id"] for s in sessions])
    if any(not s.complete for s in sessions):
        raise ValueError("incomplete/recovered sessions cannot enter baseline training")
    synthetic = {s.metadata["synthetic"] for s in sessions}
    if len(synthetic) != 1:
        raise ValueError("synthetic and real sessions cannot share a training manifest")
    if True in synthetic and not allow_synthetic:
        raise ValueError("synthetic data requires --allow-synthetic; output stays synthetic-only")
    if not all(np.isfinite(v) and v >= 0 for v in (min_duration_s, gap_merge_s)):
        raise ValueError("postprocessing durations must be finite and nonnegative")
    training_sessions = [s for s in sessions if s.metadata["participant_id"] in splits["train"]]
    signature = input_signature(training_sessions[0], task, config)
    expected_streams = {"emg"} if task == "awake" else {"audio"}
    if config["motion"]:
        expected_streams.add("motion")
    if set(signature) != expected_streams:
        raise ValueError("first training session lacks required channels")
    awake_definition = None
    reference_presence = {"awake_reference" in session.metadata for session in sessions}
    if task == "awake" and len(reference_presence) > 1:
        raise ValueError("awake cohort cannot mix specified and unspecified reference protocols")
    # Refuse hidden channel/rate changes during development rather than silently selecting easy sessions.
    for session in sessions:
        if task == "awake" and (not session.metadata["synthetic"] or "awake_reference" in session.metadata):
            validate_awake_reference(session, verify_labels=session.metadata["participant_id"] not in splits["test"])
            protocol = session.metadata["awake_reference"]
            definition = {key: protocol[key] for key in ("protocol_id", "protocol_version", "placement", "pressure_threshold_kpa", "hold_min_s", "interruption_min_s")}
            if awake_definition is not None and definition != awake_definition:
                raise ValueError("awake reference protocol/decision rule differs across cohort")
            awake_definition = definition
        acquisition = session.metadata["acquisition_config"]
        alignment = acquisition.get("reference_alignment", {})
        if not isinstance(alignment, dict):
            raise ValueError("reference_alignment must be an object")
        error = alignment.get("max_error_s")
        if (not isinstance(alignment.get("method"), str) or not alignment["method"].strip()
                or isinstance(error, bool) or not isinstance(error, (int, float)) or not np.isfinite(error)
                or error < 0 or error > config["hop_s"] / 2):
            raise ValueError("reference alignment needs a documented method and error <= half the output bin")
        if any(not isinstance(item["signal_chain"], dict) or not item["signal_chain"] for item in input_signature(session, task, config).values()):
            raise ValueError("required streams need a documented signal_chain in acquisition_config")
        if not session.metadata["synthetic"]:
            for stream in input_signature(session, task, config).values():
                chain = stream["signal_chain"]
                if not ("clip_low" in chain and "clip_high" in chain) and chain.get("clipping_mask_applied") is not True:
                    raise ValueError("real streams require declared clipping rails or an upstream clipping mask")
        if input_signature(session, task, config) != signature:
            raise ValueError("input signatures differ; prepare one explicit hardware/format cohort")
    train_rows, train_features, names = collect(sessions, splits["train"], task, config, signature)
    validation_rows, validation_features, validation_names = collect(sessions, splits["validation"], task, config, signature)
    X, y = _labelled(train_rows, train_features)
    X_validation, y_validation = _labelled(validation_rows, validation_features)
    if names != validation_names:
        raise ValueError("training and validation feature orders differ")
    model = fit_baseline(X, y, task, names, seed=seed)
    selection_probabilities = predict_proba(model, X_validation)
    selection_labels = y_validation
    if task == "awake":
        selection_labels, selection_probabilities = 1 - y_validation, 1 - selection_probabilities
    threshold, selection_score = select_threshold(selection_labels, selection_probabilities)
    model.update({
        "model_version": f"arc-{task}-{__version__}",
        "intended_use": "research_only", "synthetic_only": True in synthetic,
        "clinical_validation": "none", "live_coaching_enabled": False,
        "window_config": config, "input_signature": signature, "awake_reference_definition": awake_definition,
        "decision": {"threshold": threshold, "threshold_target": "interruption" if task == "awake" else "respiratory_event", "min_duration_s": min_duration_s, "gap_merge_s": gap_merge_s,
                     "iou_threshold": 0.3,
                     "selection": "validation frame balanced accuracy; ties nearest 0.5 then lower threshold",
                     "validation_balanced_accuracy": selection_score,
                     "event_rules": "research configuration, not clinical event criteria"},
        "splits": splits, "dataset_digest": dataset_identity(sessions),
        "provenance": [{key: s.metadata[key] for key in ("participant_id", "session_id", "source", "source_version", "permission", "label_source")}
                       for s in sessions],
        "environment": {"python": platform.python_version(), **{package: version(package) for package in ("numpy", "scipy", "scikit-learn")}},
        "training_participants": len(splits["train"]), "validation_participants": len(splits["validation"]),
    })
    model["bundle_digest"] = fingerprint(model)
    predictions = _assign(validation_rows, predict_proba(model, validation_features))
    report = score_predictions(predictions, reference_events=reference_events(sessions, splits["validation"], task), **{key: model["decision"][key] for key in ("threshold", "min_duration_s", "gap_merge_s", "iou_threshold")})
    report.update(partition="validation", synthetic_only=model["synthetic_only"], model_digest=model["bundle_digest"],
                  warning="Model selection data. Not an unbiased held-out performance estimate.")
    comparator = None
    if task == "awake":
        comparator = fit_awake_threshold(X, y, names)
        comparison_rows = _assign(validation_rows, predict_proba(comparator, validation_features))
        report["amplitude_threshold_comparator"] = score_predictions(comparison_rows, reference_events=reference_events(sessions, splits["validation"], task), threshold=0.5,
                                                                     min_duration_s=min_duration_s, gap_merge_s=gap_merge_s)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "model.json", model)
    write_json(output / "splits.json", splits)
    write_json(output / "validation.json", report)
    if comparator is not None:
        comparator.update(window_config=config, input_signature=signature, synthetic_only=model["synthetic_only"],
                          intended_use="research_only", dataset_digest=model["dataset_digest"])
        write_json(output / "amplitude-comparator.json", comparator)
    return model, report


def load_model(path):
    model = json.loads(Path(path).read_text())
    validate_bundle(model)
    if model.get("software_version") != __version__:
        raise ValueError("model software version differs; verify preprocessing parity before migration")
    digest = model.get("bundle_digest")
    if not digest or digest != fingerprint({k: v for k, v in model.items() if k != "bundle_digest"}):
        raise ValueError("model metadata/weights checksum mismatch")
    validate_config(model["task"], model["window_config"])
    if model.get("intended_use") != "research_only" or type(model.get("synthetic_only")) is not bool:
        raise ValueError("model lacks research-use provenance")
    required_streams = {"emg"} if model["task"] == "awake" else {"audio"}
    if model["window_config"]["motion"]:
        required_streams.add("motion")
    signature = model.get("input_signature")
    if not isinstance(signature, dict) or set(signature) != required_streams:
        raise ValueError("model input signature is incomplete")
    for stream in signature.values():
        if not isinstance(stream, dict):
            raise ValueError("invalid stream signature")
        rate, channels, units = stream.get("sample_rate_hz"), stream.get("channels"), stream.get("units")
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not np.isfinite(rate) or rate <= 0:
            raise ValueError("invalid stream sample rate")
        if (not isinstance(channels, list) or not channels or not isinstance(units, list) or len(channels) != len(units)
                or any(not isinstance(v, str) or not v.strip() for v in channels + units)
                or len(set(channels)) != len(channels) or not isinstance(stream.get("signal_chain"), dict)):
            raise ValueError("invalid stream channel/units/configuration contract")
    decision = model.get("decision")
    if not isinstance(decision, dict):
        raise ValueError("model requires frozen decision rules")
    for key in ("threshold", "min_duration_s", "gap_merge_s", "iou_threshold"):
        value = decision.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"invalid decision {key}")
        if key in ("threshold", "iou_threshold") and value > 1:
            raise ValueError(f"invalid decision {key}")
    expected_target = "interruption" if model["task"] == "awake" else "respiratory_event"
    if decision.get("threshold_target") != expected_target:
        raise ValueError("decision threshold target does not match task")
    for key in ("model_version", "dataset_digest"):
        if not isinstance(model.get(key), str) or not model[key].strip():
            raise ValueError(f"model requires {key}")
    provenance = model.get("provenance")
    if not isinstance(provenance, list) or not provenance or any(not isinstance(item, dict) for item in provenance):
        raise ValueError("model requires training data provenance")
    try:
        validate_splits(model.get("splits"), [item["participant_id"] for item in provenance])
    except KeyError as exc:
        raise ValueError("invalid participant provenance") from exc
    if model.get("clinical_validation") != "none" or model.get("live_coaching_enabled") is not False:
        raise ValueError("this implementation accepts research-only bundles with coaching disabled")
    return model


def predict(session, model, batch_size=512):
    """Bounded waveform reads and batched forest evaluation; no stored identity mapping."""
    validate_bundle(model)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if model["synthetic_only"] != session.metadata["synthetic"]:
        raise ValueError("synthetic-only models and real recordings cannot be mixed")
    batch, features = [], []
    begin = time.perf_counter()
    def finish():
        probabilities = predict_proba(model, np.asarray(features)) if features else []
        output = _assign(batch, probabilities)
        elapsed = (time.perf_counter() - begin) * 1000
        for row in output:
            row.update(model_version=model["model_version"], model_digest=model["bundle_digest"],
                       synthetic_only=model["synthetic_only"], intended_use="research_only",
                       recording_status=session.recording_status,
                       batch_processing_ms=elapsed,
                       probability_calibration=model["probability_calibration"],
                       latency_note="Offline batch processing only; excludes capture, transport and cue confirmation.")
        return output
    for row, vector, names in iter_windows(session, model["task"], model["window_config"], model["input_signature"]):
        if vector is not None:
            if names != model["feature_names"]:
                raise ValueError("runtime feature order differs from model bundle")
            row["feature_index"] = len(features)
            features.append(vector)
        batch.append(row)
        if len(batch) >= batch_size:
            yield from finish()
            batch, features = [], []
            begin = time.perf_counter()
    if batch:
        yield from finish()


def evaluate(sessions, model, partition="test"):
    if partition not in ("validation", "test"):
        raise ValueError("evaluation partition must be validation or test")
    if dataset_identity(sessions) != model["dataset_digest"]:
        raise ValueError("dataset changed since model freeze; use a separately versioned external evaluation")
    validate_splits(model["splits"], [s.metadata["participant_id"] for s in sessions])
    rows = []
    for session in sessions:
        if not session.complete:
            raise ValueError("incomplete/recovered sessions cannot enter held-out evaluation")
        if session.metadata["participant_id"] in model["splits"][partition]:
            if model["task"] == "awake" and (not session.metadata["synthetic"] or "awake_reference" in session.metadata):
                validate_awake_reference(session, verify_labels=True)
            rows.extend(predict(session, model))
    report = score_predictions(rows, reference_events=reference_events(sessions, model["splits"][partition], model["task"]), **{key: model["decision"][key] for key in ("threshold", "min_duration_s", "gap_merge_s", "iou_threshold")})
    report.update(partition=partition, synthetic_only=model["synthetic_only"], model_digest=model["bundle_digest"],
                  dataset_digest=model["dataset_digest"], clinical_validation="none")
    return report
