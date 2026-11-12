"""Synthetic integration fixtures. Never evidence of physiological performance."""
from pathlib import Path

import numpy as np

from .dataset import load_manifest, write_json
from .evaluation import make_splits
from .pipeline import evaluate, load_model, predict, train
from .session import write_session
from .reference import labels_from_pressure


def create_dataset(path, participants=6):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    sessions = []
    for participant in range(participants):
        rng = np.random.default_rng(100 + participant)
        duration = 10
        emg_time = np.arange(duration * 1600) / 1600
        hold = ((emg_time >= 1) & (emg_time < 3)) | ((emg_time >= 5) & (emg_time < 7))
        emg = rng.normal(size=(len(emg_time), 2)) * np.where(hold, 12, 2)[:, None]
        audio_time = np.arange(duration * 16000) / 16000
        event = ((audio_time >= 3) & (audio_time < 5)) | ((audio_time >= 7) & (audio_time < 9))
        audio = (np.sin(2 * np.pi * 220 * audio_time) * np.where(event, .01, .12)
                 + rng.normal(0, .003, len(audio_time)))[:, None]
        emg_valid = ~((emg_time >= 4) & (emg_time < 4.1))
        audio_valid = ~((audio_time >= 2) & (audio_time < 2.1))
        pressure_time = np.arange(duration * 100) / 100
        pressure_hold = ((pressure_time >= 1) & (pressure_time < 3)) | ((pressure_time >= 5) & (pressure_time < 7))
        pressure = np.where(pressure_hold, 10., 0.)[:, None]
        pressure_valid = ~((pressure_time >= 8.2) & (pressure_time < 8.3))
        protocol = {"schema_version": 1, "status": "synthetic", "stream": "reference_pressure",
                    "protocol_id": "synthetic-pressure-software-check", "protocol_version": "1",
                    "instrument_id": "generated-reference", "instrument_model": "no physical instrument",
                    "placement": "simulated tongue-palate contact; no human measurement",
                    "calibration_id": "synthetic-kpa-scale", "calibration_record": "generator constants, not hardware calibration",
                    "source_record": "reference_pressure stream in this synthetic session",
                    "approved_by": "not applicable: synthetic software fixture",
                    "approval_reference": "not clinical approval; test-only protocol",
                    "pressure_threshold_kpa": 5., "hold_min_s": .1, "interruption_min_s": .1}
        metadata = {"participant_id": f"synthetic-person-{participant:02d}", "session_id": f"synthetic-session-{participant:02d}",
                    "placement_id": "synthetic-placement", "source": "arc-selfcheck", "source_version": "2",
                    "permission": "generated synthetic fixture, no participant data", "synthetic": True,
                    "label_source": "synthetic pressure trace for awake; generated audio states for night; not clinical truth",
                    "awake_reference": protocol,
                    "acquisition_config": {"signal_chain": {"emg": {"generator": "Gaussian amplitude states"},
                                                             "audio": {"generator": "sine amplitude states", "clip_low": -1., "clip_high": 1.}},
                                           "reference_alignment": {"method": "shared synthetic sample clock", "max_error_s": 0.}}}
        streams = {
            "emg": {"sample_rate_hz": 1600, "channels": ["synthetic_a", "synthetic_b"], "units": ["uV", "uV"],
                    "samples": emg, "valid": emg_valid,
                    "calibration": {"id": "synthetic-calibration", "source": "generator constant available before recording",
                                    "rms_reference": [12., 12.], "available_at_s": 0.}},
            "audio": {"sample_rate_hz": 16000, "channels": ["mono"], "units": ["FS"], "samples": audio, "valid": audio_valid},
        }
        streams["reference_pressure"] = {"sample_rate_hz": 100, "channels": ["tongue_palate_pressure"], "units": ["kPa"],
                                         "samples": pressure, "valid": pressure_valid}
        labels = labels_from_pressure(pressure, pressure_valid, 100, 0., protocol)
        for task, spans in {
            "night_audio": [(0, 3, 0), (3, 5, 1), (5, 7, 0), (7, 9, 1), (9, 10, 0)],
        }.items():
            for start, end, state in spans:
                labels.append({"task": task, "start_s": start, "end_s": end,
                               "state": "positive" if state else "negative", "provenance": "synthetic generator"})
        folder = f"session-{participant:02d}"
        write_session(path / folder, metadata, streams, labels)
        sessions.append(folder)
    write_json(path / "manifest.json", {"schema_version": 1, "sessions": sessions})
    return path / "manifest.json"


def selfcheck(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = create_dataset(output / "synthetic-data")
    sessions = load_manifest(manifest_path)
    splits = make_splits([s.metadata["participant_id"] for s in sessions])
    write_json(output / "splits.json", splits)
    summary = {"synthetic_only": True, "clinical_performance_evidence": False, "tasks": {}}
    for task in ("awake", "night_audio"):
        train(sessions, splits, task, output / task, allow_synthetic=True)
        model = load_model(output / task / "model.json")
        report = evaluate(sessions, model)
        write_json(output / task / "test.json", report)
        replay = list(predict(sessions[0], model))
        assert len(replay) == 100 and any(not r["valid"] for r in replay)
        assert all(r["probability"] is None for r in replay if not r["valid"])
        assert all(0 <= r["probability"] <= 1 for r in replay if r["valid"])
        summary["tasks"][task] = {"bundle_loaded": True, "replay_bins": len(replay),
                                  "abstention_checked": True, "held_out_evaluation_completed": True}
    write_json(output / "selfcheck.json", summary)
    return summary
