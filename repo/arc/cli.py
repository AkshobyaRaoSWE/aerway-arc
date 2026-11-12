"""Local research workflow and bench-only BLE recording. No treatment API."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .dataset import DEFAULTS, load_manifest, write_json
from .evaluation import extract_events, make_splits
from .pipeline import evaluate, load_model, predict, train
from .session import Session, recover_session, write_session


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-session", help="verify schema and every chunk checksum")
    validate.add_argument("session")
    recover = commands.add_parser("recover-session", help="copy verified committed chunks into an incomplete recovery session")
    recover.add_argument("session")
    recover.add_argument("output")
    replay = commands.add_parser("replay-session", help="export timestamped samples, channel units and validity to CSV")
    replay.add_argument("session")
    replay.add_argument("--output", required=True)
    plot = commands.add_parser("plot-session", help="plot recorded channels with explicit missing-data shading")
    plot.add_argument("session")
    plot.add_argument("--output", required=True)
    view = commands.add_parser("view", help="local recording/replay view and optional offline scoring")
    view.add_argument("session")
    view.add_argument("--model")
    view.add_argument("--port", type=int, default=8765)
    capture = commands.add_parser("capture-ble", help="record confirmed bench BLE protocol; electrodes disconnected")
    capture.add_argument("output")
    capture.add_argument("--contract", required=True)
    capture.add_argument("--metadata", required=True)
    capture.add_argument("--address", required=True)
    capture.add_argument("--duration-s", required=True, type=float)
    reference = commands.add_parser("check-reference", help="verify awake labels against the preserved pressure reference")
    reference.add_argument("session")
    pack = commands.add_parser("pack", help="pack pre-aligned numeric NPY arrays from a JSON specification")
    pack.add_argument("specification")
    pack.add_argument("output")
    split = commands.add_parser("split", help="freeze participant groups before training")
    split.add_argument("manifest")
    split.add_argument("--output", required=True)
    split.add_argument("--seed", type=int, default=42)
    fit = commands.add_parser("train", help="fit train participants, select threshold on validation participants")
    fit.add_argument("manifest")
    fit.add_argument("--splits", required=True)
    fit.add_argument("--task", choices=list(DEFAULTS), required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--allow-synthetic", action="store_true")
    fit.add_argument("--motion", action="store_true")
    fit.add_argument("--window-s", type=float)
    fit.add_argument("--min-duration-s", type=float, default=0)
    fit.add_argument("--gap-merge-s", type=float, default=0)
    infer = commands.add_parser("predict", help="research probabilities, invalid mask, and candidate intervals")
    infer.add_argument("session")
    infer.add_argument("--model", required=True)
    infer.add_argument("--output", required=True)
    scoring = commands.add_parser("evaluate", help="score frozen model against original held-out participants")
    scoring.add_argument("manifest")
    scoring.add_argument("--model", required=True)
    scoring.add_argument("--partition", choices=["validation", "test"], default="test")
    scoring.add_argument("--output", required=True)
    check = commands.add_parser("selfcheck", help="synthetic software test, never clinical validation")
    check.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-session":
            session = Session(args.session)
            result = {"session_id": session.metadata["session_id"], "duration_s": session.duration_s,
                      "streams": list(session.streams), "digest": session.content_digest, "checksums_verified": True,
                      "recording_status": session.recording_status, "complete": session.complete}
        elif args.command == "recover-session":
            recovered = Session(recover_session(args.session, args.output))
            result = {"session": str(recovered.path), "recording_status": recovered.recording_status,
                      "duration_s": recovered.duration_s, "complete": recovered.complete}
        elif args.command == "replay-session":
            from .replay import export_replay
            result = export_replay(Session(args.session), args.output)
        elif args.command == "plot-session":
            from .replay import plot_session
            plot_session(Session(args.session), args.output)
            result = {"plot": str(Path(args.output).resolve())}
        elif args.command == "view":
            from .viewer import serve_session
            serve_session(args.session, port=args.port, model_path=args.model)
            return 0
        elif args.command == "capture-ble":
            import asyncio
            from .ble_receiver import run_ble_capture
            contract = json.loads(Path(args.contract).read_text())
            metadata = json.loads(Path(args.metadata).read_text())
            path = asyncio.run(run_ble_capture(args.output, metadata, contract, args.address,
                                              duration_s=args.duration_s))
            result = {"session": str(Path(path).resolve()), "scope": "bench_only"}
        elif args.command == "check-reference":
            from .reference import validate_awake_reference
            session = Session(args.session)
            validate_awake_reference(session, verify_labels=True)
            result = {"reference_labels_verified": True, "synthetic_only": session.metadata["synthetic"],
                      "session_id": session.metadata["session_id"], "clinical_validation": "none"}
        elif args.command == "pack":
            spec_path = Path(args.specification).resolve()
            specification = json.loads(spec_path.read_text())
            if not isinstance(specification, dict) or not isinstance(specification.get("streams"), dict) or not specification["streams"]:
                raise ValueError("pack specification needs a nonempty streams object")
            streams = specification["streams"]
            for stream in streams.values():
                if not isinstance(stream, dict):
                    raise ValueError("each stream specification must be an object")
                for name in ("samples", "valid"):
                    stream[name] = np.load(spec_path.parent / stream.pop(f"{name}_file"), allow_pickle=False, mmap_mode="r")
            destination = write_session(args.output, specification["metadata"], streams, specification["labels"])
            result = {"session": str(destination.resolve())}
        elif args.command == "split":
            sessions = load_manifest(args.manifest)
            result = make_splits([s.metadata["participant_id"] for s in sessions], seed=args.seed)
            write_json(args.output, result)
        elif args.command == "train":
            sessions = load_manifest(args.manifest)
            splits = json.loads(Path(args.splits).read_text())
            config = dict(DEFAULTS[args.task], motion=args.motion)
            if args.window_s is not None:
                config["window_s"] = args.window_s
            model, _ = train(sessions, splits, args.task, args.output, config, seed=args.seed,
                             allow_synthetic=args.allow_synthetic, min_duration_s=args.min_duration_s,
                             gap_merge_s=args.gap_merge_s)
            result = {"model": str(Path(args.output).resolve() / "model.json"), "synthetic_only": model["synthetic_only"],
                      "clinical_validation": "none"}
        elif args.command == "predict":
            model = load_model(args.model)
            session = Session(args.session)
            rows = list(predict(session, model))
            p = np.asarray([r["probability"] if r["valid"] else np.nan for r in rows])
            if model["task"] == "awake":
                p = 1 - p
            rules = model["decision"]
            events = extract_events([r["start_s"] for r in rows], [r["end_s"] for r in rows], p,
                                    np.asarray([r["valid"] for r in rows], dtype=bool), rules["threshold"],
                                    rules["min_duration_s"], rules["gap_merge_s"])
            result = {"intended_use": "research_only", "synthetic_only": model["synthetic_only"],
                      "event_target": rules["threshold_target"], "events": events, "predictions": rows}
            write_json(args.output, result)
            result = {"output": str(Path(args.output).resolve()), "prediction_bins": len(rows), "candidate_events": len(events)}
        elif args.command == "evaluate":
            result = evaluate(load_manifest(args.manifest), load_model(args.model), args.partition)
            write_json(args.output, result)
            result = {"output": str(Path(args.output).resolve()), "partition": args.partition,
                      "synthetic_only": result["synthetic_only"], "clinical_validation": "none"}
        else:
            from .selfcheck import selfcheck
            result = selfcheck(args.output)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        print(f"arc: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
