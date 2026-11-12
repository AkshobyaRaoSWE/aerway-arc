# Arc model pipelines

Executable research baselines for two separate targets:

- **Awake:** reference-defined hold probability from calibrated regional EMG; optional six-axis motion. Random forest plus amplitude-threshold comparator.
- **Night:** respiratory-event candidate probability from mono microphone audio. MFCC/energy features plus logistic regression.

Neither model is clinically trained or validated. No participant recordings were supplied. Synthetic fixtures test software behavior only. Live coaching remains disabled; outputs do not diagnose OSA, calculate clinical AHI, assess treatment benefit, or prescribe exercises.

Architecture and readiness gates: [PDF](docs/architecture.pdf) · [HTML source](docs/architecture.html).

## Run

Python 3.11+ with NumPy, SciPy and scikit-learn. From this directory:

```sh
python3 -m pip install -e '.[test,plot]'
python3 -m pytest -q
python3 -m arc selfcheck --output runs/synthetic-check
```

Selfcheck creates explicit synthetic recordings, freezes person-level splits, trains both baselines, selects thresholds using validation participants, exports JSON models, reloads them, evaluates held-out participants, and verifies abstention. Outputs refuse overwriting an existing run.

## Real recording contract

`arc.session.SessionWriter(path, metadata, stream_configs)` records incrementally. Each `append(stream, samples, valid)` synchronizes both chunk files, their directory entries, and an atomically replaced manifest checkpoint before acknowledging. `finalize(labels)` marks the session complete. `write_session(...)` uses the same core to pack existing arrays. `Session(path)` verifies hashes, schema, shapes and paths. `Session.read(name, start_s, end_s)` reads bounded windows; absent samples remain invalid. Invalid samples are never interpreted as a failed hold or an apnea.

An interrupted writer preserves every acknowledged chunk. `Session(path)` loads the last committed checkpoint; orphan/torn files are ignored. `recover_session(source, destination)` copies only verified committed data, leaves the source untouched, and marks output `recovered`/incomplete. Training and held-out evaluation refuse incomplete sessions. No automatic writer resume or hardware power-loss qualification is claimed. This is a **transport-independent incremental recorder**, not a BLE receiver. It accepts already aligned stream clocks. A transport adapter must preserve original sample counters, boot IDs, sequence gaps, raw ADC counts, channel wiring, scaling, filter settings and clock anchors in its source files/configuration. Do not align sensors or reference labels by packet arrival time. Hardware protocol, BLE UUIDs and effective ADC configuration remain unprovided.

Metadata fields:

| Field | Meaning |
|---|---|
| `participant_id` | Stable research ID across every session, reattachment and night. Resolve aliases across datasets before splitting. |
| `session_id`, `placement_id` | Unique recording ID; electrode reattachment ID. |
| `source`, `source_version` | Dataset/device release identity. |
| `permission` | Data-use approval/license reference and permitted scope. A populated string does not verify permission. |
| `synthetic` | Explicit boolean. Synthetic and real data cannot share a run or model inference path. |
| `label_source` | Independent reference and annotation provenance. Task instructions are not ground truth. |
| `acquisition_config.signal_chain` | Per-stream hardware/filter/scaling configuration. Included in the model input signature. |
| `acquisition_config.reference_alignment` | `method` and measured `max_error_s`. Training currently requires error no greater than half the 100 ms output bin. This is an engineering alignment gate, not a clinical cutoff. |

Each stream provides `sample_rate_hz`, unique `channels`, matching `units`, `start_time_s`, numeric `samples[N,C]`, and boolean `valid[N]`. Use `emg` in `uV`, `audio` at exactly 16000 Hz/mono in normalized `FS`, optional `motion` ordered ax/ay/az/gx/gy/gz in m/s^2 then rad/s. Importers must explicitly resample other audio rates and preserve transformation provenance. This package never silently resamples.

Store raw ADC counts as an additional stream if converting to microvolts. For real training data, each required stream's `signal_chain` must declare `clip_low`/`clip_high` in its stored units, or `clipping_mask_applied: true` with a trustworthy upstream mask. Rails can be scalar or per-channel lists. Positive int16 saturation normalized by 32768 is **32767/32768**, not 1. Lead-off, packet loss, unusable contact/motion and microphone disconnection require acquisition-quality evidence; a quiet waveform alone cannot establish them.

EMG additionally requires stream-level `calibration`:

```json
{"id":"calibration-record-id","source":"documented deployment calibration", "rms_reference":[12.0,13.0], "available_at_s":0.0}
```

Values above illustrate shape only. Use independently recorded channel RMS references, never future trial statistics or held-out labels. Calibration must be available before the entire feature window. Missing calibration abstains. No nominal EMG bandpass is imposed before ADC bandwidth is confirmed.

Labels are half-open, session-clock intervals:

```json
{"task":"awake","start_s":2.03,"end_s":4.87,"state":"positive","provenance":"reference annotation record","confounds":[]}
```

`positive` means maintained hold for awake, reference respiratory event for night. `negative` means independently scored absence of that target. `exclude` masks unusable/unscored regions. Preserve original respiratory subtypes, treatment portions, scoring protocol and annotator identity in source annotations. Wake, titration and unwanted noise need explicit policy/exclusion. Unannotated time never becomes a negative label. Only nonoverlapping scored labels are accepted; exclusions may overlap. Boundary bins with mixed labels are excluded from model fitting, while exact original event boundaries are retained for event evaluation.

To pack numeric files without writing a custom Python loader, create a JSON specification containing `metadata`, `labels`, and `streams`, replacing each stream's arrays with `samples_file` and `valid_file` paths relative to the specification. Then:

```sh
python3 -m arc pack local-data/session-spec.json local-data/session-001
python3 -m arc validate-session local-data/session-001
```

NPY loading disables pickle. Inputs must already be numeric and synchronized. A schema-valid file is not proof of correct hardware units, permissions, clock alignment, or independent labels.

## Concrete awake reference

Selected reference: **JMS TPM-02 with manufacturer pressure-recording software**, probe and connecting tube. The [official software manual, printed p17/Figure39](https://orarize.com/english/zetsuatsu/download/instruction_for_use.pdf#page=19) shows measurement-time / pressure CSV with 0.05-second increments in its example. This is documentation evidence, not measured cadence or verified Arc synchronization. Selection and acceptance checks: [reference decision](docs/reference-selection.pdf).

Store its calibrated, clock-aligned pressure as a **separate** `reference_pressure` stream, channel `tongue_palate_pressure`, units `kPa`. It never enters the EMG feature vector. Every trial reset needs an explicit session-time offset; missing intervals need false validity. Instrument/display arrival time is not a substitute for measured synchronization.

`arc.reference.labels_from_pressure(...)` implements an explicit offline rule:

- Valid pressure at/above the protocol threshold for at least `hold_min_s`: positive maintained-hold label, from run onset.
- Valid pressure below threshold for at least `interruption_min_s`: negative interruption label, from run onset.
- Shorter runs and invalid reference samples: exclude. Never bridge invalid samples. Add explicit exclusions outside the defined practice task.

Real awake sessions require `metadata.awake_reference`: protocol ID/version, instrument ID/model, placement, calibration ID/record, source record, approval identity/reference, numerical pressure/duration rules, `stream: reference_pressure` and `status: approved`. The converter adds the protocol digest to labels. Training reproduces train/validation labels from pressure and checks one shared rule across participants; held-out pressure is checked only during evaluation. The software validates traceability, **not authenticity** of approval or physical independence.

No clinical threshold, placement approval or instrument calibration has been invented. The included **5 kPa / 0.1 s** settings are arbitrary synthetic software-test values, never a protocol for people. Until an approved real protocol and measured synchronization are supplied, real awake training is blocked.

## Reviewable saved-session evidence

```sh
python3 scripts/build_handoff.py --output deliverables/arc-review
```

Produces a source snapshot, ten synthetic participants, a held-out sample, all-channel PNG, exact CSV replay, both baseline runs, a killed-process/recovery example, test logs, checksums and one ZIP. `scripts/verify_handoff.py` verifies the extracted package independently of the original workspace. This is software evidence, not real-data model performance.

```sh
python3 -m arc validate-session SESSION
python3 -m arc check-reference SESSION
python3 -m arc plot-session SESSION --output session.png
python3 -m arc replay-session SESSION --output replay
python3 -m arc recover-session INTERRUPTED_SESSION RECOVERED_SESSION
```

The PNG labels channels, units, session seconds and invalid spans. CSV retains every numeric value plus sample index, timestamp and validity; the audit records exact gap ranges. Replay is deterministic offline replay, not a phone or real-time Bluetooth demonstration.

## Train and evaluate

Create a dataset manifest using paths relative to its location:

```json
{"schema_version":1,"sessions":["session-001","session-002","session-003"]}
```

At least three distinct participants are required for nonempty train/validation/test groups. This software minimum is not a sufficient cohort size for validation. Both classes must occur in valid training and validation windows.

```sh
python3 -m arc split local-data/manifest.json --output local-data/splits.json
python3 -m arc train local-data/manifest.json --splits local-data/splits.json --task awake --output runs/awake
python3 -m arc train local-data/manifest.json --splits local-data/splits.json --task night_audio --output runs/night
python3 -m arc evaluate local-data/manifest.json --model runs/awake/model.json --output runs/awake/test.json
python3 -m arc predict local-data/session-001 --model runs/awake/model.json --output runs/awake/replay.json
```

`--motion` adds awake motion features. `--window-s` changes feature context for a predeclared validation experiment. Defaults: awake 0.5 s context; night 2 s context; both 100 ms output bins. `--min-duration-s` and `--gap-merge-s` set research postprocessing rules before testing. Defaults are zero, deliberately not invented clinical event criteria. Compare variants on validation participants, freeze the chosen bundle, then run the test once. Do not repeatedly select architectures using test results.

A run saves model weights, feature order, preprocessing/window configuration, signal-chain signature, threshold target, frozen splits, dataset hash, provenance, software version and validation report. JSON export avoids loading executable pickle files. Hashes detect accidental changes; they do not authenticate a malicious author. Files must remain immutable while being read.

Thresholds maximize validation frame balanced accuracy over 0.05 to 0.95. Awake thresholds operate on **1 - hold probability**, consistently with interruption event extraction. Forest/logistic outputs are uncalibrated probabilities, not calibrated clinical confidence. The amplitude comparator emits a hard binary score. No coaching cue policy is enabled.

Evaluation reports event sensitivity, precision, false detections per recording hour, duration-weighted frame PR/Brier, onset/offset error, overlap, usable coverage, censored events and participant-bootstrap intervals. An event match is one-to-one at an explicitly reported IoU threshold of 0.3. Unknown/invalid gaps never merge events. A partially unobservable reference event is censored, not converted into several easier events. Event counts and exposure exclude censored spans together; frame coverage is reported separately. Insufficient exposure is explicit, and undefined rates are null. Participant intervals are unavailable with fewer than two test participants.

Counts per recording hour are not clinical AHI. False interruption counts are not actual coaching cues; instructed-practice-only cue rates require a separately implemented practice/cue protocol. Confound summaries are frame-level because fragmenting event timelines would bias event counts.

## What remains before deployment

1. Firmware/channel/ADC/quality-mask contract and clock-drift measurements; board electrical review; permitted bench capture. No body-connected acquisition or stimulation interface is implemented.
2. Independently labelled, permitted recordings. Awake needs the actual placement/task plus reference. Night needs synchronized acoustic/reference events and explicit scoring coverage. No external dataset permission is assumed or download performed.
3. Prespecified clinician-approved label definition and acceptance targets, adequate participant sample size, locked internal and external validation, and Arc-hardware transfer evaluation. Current CLI scores the original frozen cohort; an external cohort requires a separately versioned evaluation protocol.
4. Probability calibration and any cue threshold/dwell/cooldown policy, chosen without test leakage. Then later-session personalization and phone export/replay parity with measured end-to-end latency.
5. A compact log-mel CNN + bidirectional GRU with 20 s context is a **future offline comparison**, not implemented or justified as superior. Build it only after permitted data and baseline errors show what needs improvement.

No cloud service, phone runtime, neural placeholder, simulated clinical claim, or automatic exercise recommendation is included.
