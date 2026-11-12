"""Deliberately kill a synthetic recorder after two acknowledged chunks.

Run only as a child process. Expected exit is SIGKILL, not success.
"""
import os
from pathlib import Path
import signal
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arc.session import SessionWriter


def main(path):
    metadata = {"participant_id": "synthetic-crash-person", "session_id": "synthetic-crash-session",
                "placement_id": "generated", "source": "crash-probe", "source_version": "1",
                "permission": "generated software fixture", "synthetic": True,
                "label_source": "no reference labels", "acquisition_config": {"clock": "synthetic100Hz"}}
    writer = SessionWriter(path, metadata, {"emg": {"sample_rate_hz": 100, "channels": ["synthetic_counter"], "units": ["uV"]}})
    writer.append("emg", np.arange(100, dtype=float)[:, None], np.ones(100, dtype=bool))
    valid = np.ones(100, dtype=bool)
    valid[30:35] = False
    writer.append("emg", np.arange(100, 200, dtype=float)[:, None], valid)
    with (Path(path) / "uncommitted-torn.npy").open("wb") as stream:
        stream.write(b"incomplete array; never committed")
        stream.flush()
        os.fsync(stream.fileno())
    print("Two acknowledged chunks: 200 samples. Sending SIGKILL with an orphan partial file present.", flush=True)
    os.kill(os.getpid(), signal.SIGKILL)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
