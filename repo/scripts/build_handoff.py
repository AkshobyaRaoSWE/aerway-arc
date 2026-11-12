"""Build inspectable source, saved-session, replay, baseline and crash evidence."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from arc.dataset import load_manifest, write_json
from arc.evaluation import make_splits
from arc.pipeline import evaluate, load_model, predict, train
from arc.reference import validate_awake_reference
from arc.replay import export_replay, plot_session
from arc.selfcheck import create_dataset
from arc.session import Session, recover_session
from verify_handoff import verify


def build(destination):
    if not __debug__:
        raise RuntimeError("handoff checks require assertions enabled")
    destination = Path(destination).resolve()
    archive = destination.with_suffix(".zip")
    if destination.exists() or archive.exists():
        raise FileExistsError("handoff directory/archive exists; choose a new destination")
    destination.mkdir(parents=True)
    evidence = destination / "evidence"
    evidence.mkdir()
    manifest = create_dataset(evidence / "dataset", participants=10)
    sessions = load_manifest(manifest)
    splits = make_splits([s.metadata["participant_id"] for s in sessions])
    write_json(evidence / "splits.json", splits)
    selected = next(s for s in sessions if s.metadata["participant_id"] == splits["test"][0])
    shutil.copytree(selected.path, evidence / "sample-session")
    sample = Session(evidence / "sample-session")
    validate_awake_reference(sample, verify_labels=True)
    export_replay(sample, evidence / "replay")
    plot_session(sample, evidence / "sample-session.png")
    for task in ("awake", "night_audio"):
        base = evidence / "baselines" / task
        train(sessions, splits, task, base, allow_synthetic=True)
        model = load_model(base / "model.json")
        write_json(base / "test.json", evaluate(sessions, model))
        write_json(base / "sample-predictions.json", list(predict(sample, model)))
    crash = evidence / "crash"
    crash.mkdir()
    process = subprocess.run([sys.executable, str(SOURCE / "scripts/crash_probe.py"), str(crash / "interrupted")],
                             cwd=SOURCE, capture_output=True, text=True)
    (crash / "process.log").write_text(process.stdout + process.stderr)
    if process.returncode != -signal.SIGKILL:
        raise RuntimeError(f"crash probe did not die by SIGKILL: {process.returncode}")
    original = Session(crash / "interrupted")
    recovered = Session(recover_session(original.path, crash / "recovered"))
    values, valid = recovered.read("emg", 0, 2)
    np.testing.assert_array_equal(values[:, 0], np.arange(200))
    assert valid.sum() == 195 and not recovered.complete
    write_json(crash / "recovery-proof.json", {"process_returncode": process.returncode, "termination": "SIGKILL",
               "acknowledged_chunks": 2, "committed_samples": 200, "recovered_samples": len(values),
               "source_manifest_sha256": original.content_digest, "source_unchanged": Session(original.path).content_digest == original.content_digest,
               "invalid_samples_preserved": int((~valid).sum()), "orphan_excluded": not (recovered.path / "uncommitted-torn.npy").exists(),
               "recovered_status": recovered.recording_status, "complete": recovered.complete,
               "scope": "process-kill software test; not hardware power-loss qualification"})
    export_replay(recovered, crash / "replay")
    plot_session(recovered, crash / "recovered.png")
    checks = destination / "checks"
    checks.mkdir()
    process = subprocess.run([sys.executable, "-m", "pytest", "-q", f"--junitxml={checks / 'tests.xml'}"],
                             cwd=SOURCE, capture_output=True, text=True)
    (checks / "tests.log").write_text(process.stdout + process.stderr)
    if process.returncode:
        raise RuntimeError("test suite failed; see checks/tests.log")
    test_count = sum(int(item.get("tests", "0")) for item in ET.parse(checks / "tests.xml").getroot().iter("testsuite"))
    packages = {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "matplotlib", "pytest", "joblib", "threadpoolctl")}
    (destination / "requirements-tested.txt").write_text("\n".join(f"{name}=={value}" for name, value in packages.items()) + "\n")
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=SOURCE, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=SOURCE, capture_output=True, text=True)
    write_json(destination / "environment.json", {"python": platform.python_version(), "platform": platform.platform(),
               "source_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None, "packages": packages,
               "source_commit": git.stdout.strip() if git.returncode == 0 else None, "tests_passed": test_count})
    repo = destination / "repo"
    repo.mkdir()
    for name in ("arc", "tests", "scripts", "docs"):
        shutil.copytree(SOURCE / name, repo / name, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    for name in ("README.md", "pyproject.toml", ".gitignore"):
        shutil.copy2(SOURCE / name, repo / name)
    (destination / "README.md").write_text('''# Arc implementation review bundle

All recordings and model outputs here are SYNTHETIC SOFTWARE EVIDENCE. No human data, clinical accuracy or completed hardware validation is claimed.

## Inspect

- `repo/`: complete standalone source snapshot, tests and updated architecture.
- `evidence/sample-session/session.json`: one complete held-out sample with EMG, microphone and separate pressure reference.
- `evidence/sample-session.png`: recorded channels, actual session seconds, units and hatched invalid ranges.
- `evidence/replay/`: exact sample CSV files and channel/time/validity audit.
- `evidence/baselines/awake/` and `night_audio/`: fitted JSON models, frozen splits, validation report, held-out report and sample predictions.
- `evidence/dataset/`: all ten synthetic participants, making both baseline runs reproducible.
- `evidence/crash/`: interrupted source, recovered session, replay, plot and real SIGKILL process proof.
- `checks/`: executed tests and machine-readable JUnit results.
- `repo/docs/reference-selection.html`: selected real instrument and unresolved synchronization/clinical-criteria gates.

## Verify after extracting

Recorder/crash evidence was tested on the POSIX platform listed in environment.json. Windows recorder behavior is not qualified.

From this directory, with the tested Python version and listed dependencies:

```sh
python3 -m pip install -r requirements-tested.txt
python3 -m pip install --no-deps -e repo
python3 repo/scripts/verify_handoff.py .
python3 -m arc validate-session evidence/sample-session
python3 -m arc check-reference evidence/sample-session
python3 -m arc replay-session evidence/sample-session --output replay-again
python3 -m arc plot-session evidence/sample-session --output plot-again.png
python3 -m arc predict evidence/sample-session --model evidence/baselines/awake/model.json --output predict-again.json
python3 -m arc evaluate evidence/dataset/manifest.json --model evidence/baselines/awake/model.json --output test-again.json
```

Reproduce fitting using frozen participant IDs:

```sh
python3 -m arc train evidence/dataset/manifest.json --splits evidence/splits.json --task awake --allow-synthetic --output awake-again
python3 -m arc train evidence/dataset/manifest.json --splits evidence/splits.json --task night_audio --allow-synthetic --output night-again
```

The verifier checks archive-file hashes, exact CSV values/timestamps/validity, pressure-reference conversion, model replay, fresh baseline refitting, held-out report reproduction and recovered crash samples. Numeric model/report comparisons permit 1e-10 relative / 1e-12 absolute floating-point tolerance; structural fields stay exact. Assertions must remain enabled. Processing durations are measured again and intentionally are not compared bit-for-bit.

Crash guarantee: append acknowledges only after data and checkpoint sync. A failed in-flight append can leave orphan files; recovery ignores them. Earlier committed chunks remain readable. Recovered sessions remain incomplete and cannot enter model training/evaluation. Actual power failure, filesystem durability and BLE behavior still require hardware tests.

Actual awake reference selected: JMS TPM-02 plus manufacturer recording software. Real thresholds, placement and approval have not been fabricated. The synthetic 5 kPa / 0.1 s values only exercise code; they are not recommendations for people.
''')
    files = sorted(path for path in destination.rglob("*") if path.is_file())
    write_json(destination / "checksums.json", {str(path.relative_to(destination)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files})
    result = verify(destination)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                handle.write(path, str(path.relative_to(destination)))
    return {"directory": str(destination), "archive": str(archive), "tests_passed": test_count, **result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(build(parser.parse_args().output), indent=2))
