# Bench packet contract: draft v1

This is a proposed receiver contract, not an agreed firmware protocol. No UUID,
channel map, sample rate, calibration or hardware test result has been supplied.
`examples/bench-contract.json` is deliberately incomplete and cannot start capture.
Aarav must confirm hardware fields; Sathvik must confirm receiver interoperability.
Those are pending review assignments, not claims either person approved this file.

## Capture gate

Real BLE capture requires `status: confirmed`, nonempty `hardware_confirmed_by`,
`receiver_confirmed_by`, `agreement_record`, actual `service_uuid` and `notify_uuid`,
and complete stream definitions. These fields record human attestation; they do
not authenticate firmware or prove calibration. Capture metadata must explicitly
set `synthetic: false`, `bench_only: true`, and
`body_electrodes_disconnected: true`. USB is bench-only with body electrodes
disconnected. No live coaching or physical output is implemented.

Each stream specifies `id` (uint16), `name` (simple unique identifier),
`sample_rate_hz`, `start_time_s` (session-relative seconds), `dtype_code`,
ordered `channels`, per-channel `units`, `scale`, `offset`, and `max_gap_s`.
Physical values are `raw * scale + offset`; original signed integer counts are
stored in `<name>_raw`. Audio uses integer PCM plus explicit normalization in its
physical stream; IMU uses explicit axis order and units. No ADC counts are silently
renamed volts. Optional session-format `calibration` is preserved.

`signal_chain` records actual electrode/channel mapping, ADC range/resolution,
AFE gain, analog bandwidth/antialias filtering, digital filtering/decimation,
calibration evidence, and hardware flag sources. Audio and IMU record their
corresponding hardware configuration. These details require human review against
hardware, beyond schema validation. `clock` requires `tick_hz`, `first_tick`,
`first_sample: 0`, and `mapping_evidence`. All stream origins must refer to the
same device session timebase. Agree actual clock reset and wrapping behavior.
Counter wrapping requires a new session.

## Binary notification stream

Little-endian, packed header Python format `<4sIBBHI16s16s32sQQIH`:

| Field | Bytes | Meaning |
|---|---:|---|
| magic | 4 | ASCII ARC1 |
| total_bytes | 4 | Entire frame, including checksum |
| version | 1 | 1 |
| dtype | 1 | 1=int16, 2=signed int24, 3=int32 |
| stream_id | 2 | Agreed stream identifier |
| sequence | 4 | Per-stream frame counter, begins zero |
| boot_id | 16 | UUID bytes; new UUID on device reboot |
| device_session_id | 16 | UUID bytes; new UUID on acquisition start |
| configuration_sha256 | 32 | Provisioned shared configuration hash |
| first_sample | 8 | Per-stream sample counter, begins zero |
| first_device_tick | 8 | Device timestamp for first sample |
| sample_count | 4 | 1..65536 |
| channel_count | 2 | 1..32, matches stream configuration |

Payload: sample-major interleaved signed integers, then one uint8 quality flag
per sample. Signed int24 is two's-complement, low byte first. Final 4 bytes are
IEEE CRC32 of header plus payload, unsigned little-endian. Maximum complete frame
size is 1 MiB. CRC detects corruption; it is not authentication. Flags are bit0
clipping, bit1 lead-off, bit2 bad contact, bit3 device-declared invalid/gap sample.
Any set bit marks that sample invalid across channels. Unknown bits are rejected.
Raw transport preserves all flags; session valid masks preserve invalidity.

Notifications may fragment frames or contain multiple frames. The decoder uses
length and checksum, never guesses alignment after corruption. Input and pending
fragments are bounded; callback queue defaults to 128 notifications. Overflow is
fatal and leaves the session incomplete. Pick rates/packet sizes only after BLE
throughput measurements; the proposal does not establish sustainable bandwidth.

Compute configuration hash with `configuration_digest(contract)`: SHA256 over
UTF-8 JSON of `configuration`, sorted keys, compact separators, no NaN. Provision
those exact bytes/hash into firmware during agreement; do not independently
reformat JSON across languages. Signoff and transport UUID fields are outside
the hashed acquisition configuration. Frame hashes and raw `transport.bin` are
recorded for replay/audit.

## Time, loss, recovery

Timestamps derive from counters, configured sample rates and device clock anchors,
never laptop callback arrival times. For each frame, `(device_tick - first_tick) /
tick_hz` must match `first_sample / sample_rate_hz` within 50 ms. This checks
internal consistency only. Metadata explicitly says independent clock alignment
is unverified. Shared-reference bench measurements must establish actual
cross-stream alignment and drift; synthetic clock checks do not establish that.

Missing frame sequence and sample-counter ranges must agree. Every accepted loss
is logged as a half-open sample/time range; physical values become NaN and masks
false. Raw companions use zero with false validity, never fabricated valid counts.
A configured maximum gap bounds allocation. Exact repetition of the most recent
frame is logged and ignored. Other rollback/reordering, changed boot/session,
changed config, impossible clock mapping and corrupt frames fail closed: start a
new recording. No data is silently reordered or counted twice.

`SessionWriter` publishes durable incremental manifests. Completed chunks remain
readable after disconnect, cancellation, overflow or crash with incomplete status.
Raw and physical streams commit separately; an interrupted frame may leave unequal
stream tails. Incomplete sessions must be inspected/recovered before analysis.
`transport.bin` contains original incoming bytes, including rejected input;
`acquisition.jsonl` records accepted frames, losses, duplicates and explicit failure.

Stop finalizes the **locally received recording window** only. Firmware start/stop
acknowledgements and final device counters are not available in this proposal.
Therefore complete local session status does not establish absence of unobserved
tail loss or complete device-duration capture. Metadata marks that limitation.
Agree and implement firmware acknowledgements before claiming that acceptance
check passed. Unexpected disconnect never finalizes a session. Stopping with a
partial frame fails rather than silently discarding it.

## APIs and checks

`BenchRecorder(path, metadata, contract).feed(notification_bytes)` incrementally
commits data. `.stop()` returns the complete local session path; `.abort(reason)`
leaves it incomplete. Offline byte replay uses the same `.feed()` decoder and
writer, without requiring BLE. Synthetic fixtures may use draft contracts and
must retain `synthetic: true`.

`await run_ble_capture(path, metadata, contract, device_address,
duration_s=60)` subscribes to the confirmed notification characteristic. An
`asyncio.Event` passed as `stop_event` also supports user stop. No assumed control
characteristic, start command or generic UUID is sent. Firmware must implement the
agreed subscription behavior before this receiver can operate.

Software tests cover signed int24, fragmentation/coalescing, raw/scaled roundtrip,
flags, missing ranges, duplicates, reset/config/CRC/clock failures, incomplete
recovery, callback overflow/disconnect, and one minute of counter continuity with
byte replay. They are transport checks, not hardware, electrode safety, radio
throughput, calibrated-clock or model-performance evidence. Bleak integration
uses its [official client API](https://bleak.readthedocs.io/en/latest/api/client.html)
with pinned `bleak==3.1.0`; actual adapter interoperability remains untested.

Metadata shape for a real disconnected bench session (fill actual acquisition
provenance and unique IDs; these example identifiers are not hardware evidence):

```json
{
  "participant_id": "bench-no-person",
  "session_id": "unique-bench-session-id",
  "placement_id": "body-electrodes-disconnected",
  "source": "BLE bench signal input",
  "source_version": "actual firmware revision",
  "permission": "owned bench hardware",
  "synthetic": false,
  "acquisition_config": {},
  "label_source": "none",
  "bench_only": true,
  "body_electrodes_disconnected": true
}
```

Gap filling has an additional absolute bound of 1,000,000 channel-values per gap,
regardless of configured rate or tolerance. Larger gaps fail closed without an
unbounded allocation.
