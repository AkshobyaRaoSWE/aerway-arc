# Aerway Arc: recording, data and model plan

8 November 2026. Engineering proposal based on Aarav's emailed parts list, current repository files, correspondence and primary documentation. This is not a claim that sensing or clinical performance has been validated.

## Decision

Build one reliable recording pipeline first. Develop two separate analyses from it: awake exercise measurement and overnight audio research. Use a laptop for the initial logger and models. Add a phone logger next. A cloud VM is optional, not required to make the wearable work.

Do not make next-day exercise prescriptions depend on an unvalidated overnight score. Exercise instructions remain clinician-selected. A successful awake classifier does not establish a reduction in apneas.

## What Aarav actually has

His email, “Aerway / Conrad — full prototype parts list and PCB hold,” identifies the ProtoCentral ADS1293 v2 PC-4129, XIAO nRF52840 Sense, two three-lead cables, disposable electrodes, and a candidate protected 500 mAh LiPo. Exact electrode mapping, battery endurance and skin retention remain unverified.

The XIAO Sense includes a PDM microphone, IMU and BLE. Its small onboard flash is not a full-night raw recording store. There is no microSD module, SpO2 sensor or independent tongue-pressure instrument in the supplied list. [Seeed documentation](https://wiki.seeedstudio.com/XIAO_BLE/)

Aarav explicitly reports that Rev D's battery-contact pads overlap by about 0.295 mm and has placed fabrication on hold. This is his reported finding, not a new independent board inspection. Keep that hold until the geometry is corrected and native KiCad DRC and electrical review pass. A 500 mAh label alone does not establish overnight runtime.

## Data path

```text
Skin electrodes -> ADS1293 -> SPI -----+
XIAO Sense microphone -> PDM capture --+-> timed sample blocks -> BLE -> laptop/phone
XIAO Sense IMU -> motion samples ------+                              |
                                                     raw session files + gap log
                                                                    |
                                            preprocessing identical to training
                                                     /                    \
                                            awake model              audio model
                                            practice feedback        offline research report
```

SPI and PDM are local connections inside the wearable. BLE carries data to the nearby phone or computer. The phone's Wi-Fi/mobile connection can later upload a saved session. The XIAO is not independently connected to Fly.io or the internet through this design.

For the first working version, use the laptop as the BLE receiver and save data locally. This verifies the firmware and file format before mobile background operation adds another source of failures. A USB data link is appropriate for bench signals with body electrodes disconnected. It is not the proposed body-connected acquisition path.

Next implement the phone logger on one known phone model and OS. Test screen locking, app backgrounding, Bluetooth interruptions, incoming calls, low battery and full storage. Android and iOS have different background-execution behavior. A browser-based demo is not evidence of reliable overnight phone recording.

## Firmware-to-software contract

Aarav and the software team must agree on this before either side finishes implementation:

- Session and boot identifiers, firmware version and configuration hash.
- Stream identifier: ADC, microphone, IMU or device status.
- Per-stream sequence number, first-sample counter, sample count and device timestamp.
- Actual sample rate, channel order, data format, endianness and raw ADC scaling rules.
- ADC input pair for every channel, filter settings, reference/bias configuration and electrode-placement record. Five skin contacts do not automatically mean five signals or three independent muscle recordings.
- Clipping, contact/lead-off flags where supported and verified, dropped-block counters and battery status.
- Start, stop and configuration acknowledgments. No stimulation command in this acquisition interface.

Use binary blocks, not a JSON object for every sample. Capture samples into buffers and transmit separately so BLE cannot stall acquisition. Treat the ADC and audio clocks as separate timing sources even though one MCU receives them. Record counters and periodic time anchors, then measure drift rather than aligning streams by phone arrival time.

Log missing ranges explicitly. Do not replace a disconnected microphone with zeroes and label the silence an apnea. Inference must abstain when a required input is unusable. Without extra storage, long disconnections lose data. A short ring buffer cannot promise overnight recovery.

## Rates and storage: provisional engineering examples

The ADS1293 has configurable output rates and bandwidths. Rate selection changes noise and usable bandwidth, and the ECG example configuration is not automatically suitable for regional EMG. For example, TI lists an ECG-channel configuration at 1,600 samples/s with approximately 325 Hz bandwidth and another at 3,200 samples/s with approximately 640 Hz bandwidth. Select and bench-measure the intended configuration, not just its headline sample rate. [TI datasheet, Tables 10–11](https://www.ti.com/lit/ds/symlink/ads1293.pdf)

For sizing only, assume three channels, packed 24-bit samples, mono 16 kHz / 16-bit PCM audio, and a six-axis IMU at 100 Hz / 16 bits per axis:

| Stream | Payload rate |
|---|---:|
| Three ADC channels at 1,600 samples/s | 14,400 bytes/s |
| Audio | 32,000 bytes/s |
| Six-axis IMU | 1,200 bytes/s |
| Total before transport/file overhead | 47,600 bytes/s |
| Eight-hour total | 1.37 GB (decimal) |

At 3,200 ADC samples/s, this becomes 62,000 bytes/s and 1.79 GB over eight hours. This excludes timestamps, headers, retries and filesystem overhead. It is a capacity estimate, not a measured BLE throughput result. The number of actually independent ADC channels still depends on wiring.

Preserve raw audio in permitted development recordings so alternative features can be compared. Do not assume a 10 Hz loudness envelope preserves all apnea information. Raw speech can identify people, so retain only what the approved data-use arrangement allows. Later compare raw, spectral-feature and compressed representations empirically before choosing a production format. The initial 16 kHz audio proposal must also be tested against the acoustic bandwidth needed for the task.

Recommended session folder: `session.json`, chunked raw ADC data, `audio.wav` chunks, IMU samples, `events.csv`, `quality.csv`, and a checksummed manifest. Keep patient identity mappings separate from research IDs. Files should survive a crash up to the last completed chunk. Synchronize independent reference sensors explicitly.

## Model A: awake exercise measurement

Start with a narrowly defined clinician-taught task. A candidate is a sustained tongue-to-palate pressure hold, with an independently specified interruption criterion. Regional EMG can be an input to that experiment. It is not itself proof of tongue position, correct performance or therapeutic value.

Training needs paired Arc recordings and a clinician-approved independent reference, such as a suitable tongue-pressure system. Pressure alone still does not prove every aspect of technique. Include clinician observation where needed, and label the precise feature being tested rather than an undefined “correct exercise.”

Collect labelled examples of the intended task, rest, swallowing, jaw clenching, talking, head movement and poor contact under an approved protocol. Repeat sessions after electrode removal and replacement. Do not use only one person's repetitions. Existing clinical exercise recordings may help, but check placement, task, reference instrument and synchronization before treating them as transferable labels.

Begin with a transparent feature baseline, such as calibrated EMG amplitude and timing plus motion features, then logistic regression or a small tree model. Compare it against a simple threshold baseline. Test whether audio adds information rather than automatically including it. A larger neural network is justified only if it improves held-out performance and maintains useful latency.

Evaluate interruption sensitivity, false cues per minute, onset/offset timing error, abstention coverage and performance after reattachment. Report participant-level uncertainty. A later comparison must establish whether feedback helps practice more than the same instructions without sensor feedback.

## Model B: nighttime audio research

The target should first be the detection of reference-labelled respiratory-event intervals from audio, not a clinical claim of nightly OSA improvement. Audio recordings without synchronized clinical labels do not provide apnea ground truth. Snoring examples can train a snore detector, but snoring is not an apnea label.

Start with public reference data and offline experiments. Train a modest log-mel spectrogram model or audio-feature baseline before an LSTM/Transformer. Use only channels the deployed Arc will actually supply as model inputs. Clinical airflow, effort, oxygen and sleep-stage signals can establish reference labels without becoming hidden runtime inputs.

Split by person before creating training windows. Keep repeated nights together. Fit normalization and thresholds on training/validation data only. Lock the model before testing a different dataset or clinic. Report event-level sensitivity, false detections per hour, precision-recall, temporal overlap, rejected-recording rate and participant-level intervals. Accuracy alone can hide missed events and class imbalance.

A rate per recording hour is not automatically an AHI, which needs sleep time and defined apnea/hypopnea events. Longitudinal improvement needs repeatable performance across nights and independent clinical comparison. A quiet room, detached microphone or partner's snoring must not become an apparent treatment effect.

## Existing datasets and models

| Resource | What it provides | Fit and limitation |
|---|---|---|
| [PSG-Audio original paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8333307/) and [original repository](https://www.scidb.cn/en/detail?dataSetId=778740145531650048) | Original report describes 212 polysomnograms with tracheal/ambient audio and scored events. | Strong candidate for audio-method development. Confirm actual release, file availability, license, synchronization corrections and diagnostic versus CPAP-titration portions. Repository metadata was not readable through the current web tool, so download and current license are not verified. |
| [APSAA official Zenodo record](https://zenodo.org/records/14096541) | 32 subjects with audio, respiratory polygraph signals and manual event annotations. WAV/CSV files, listed archive 3.9 GB. | Useful candidate for an independent audio benchmark. The record restricts use to academic/non-commercial research and prohibits redistribution without permission. Ask the authors whether Aerway's independent venture use is allowed before adopting it. Polygraphy is not the same reference as full PSG. |
| [Apnea-ECG](https://physionet.org/content/apnea-ecg/1.0.0/) | ECG with apnea annotations. | Useful for ECG-method education. It does not supply the microphone training pairs Arc needs. |
| [YAMNet](https://www.tensorflow.org/hub/tutorials/yamnet) | Available general audio-event classifier and embeddings. | Can provide initial audio features. Its general event predictions are not an apnea detector or validated Arc output. |
| [ApneaWhisper paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12506788/) | Describes a pretrained-encoder approach to respiratory-event segmentation. | A research reference, not a verified ready-to-run Arc model. No downloadable task checkpoint was verified in this review. Its reported cohort description differs from the original PSG-Audio paper, so resolve the exact release/split before relying on its performance. |

I did not verify an open dataset matching Arc's exact awake electrode layout, task and independent exercise-quality labels. That is a search finding, not proof that no such dataset exists. Public audio datasets can start model development. They cannot establish that the under-chin hardware captures the same information.

## Where compute runs

Training means fitting the model from labelled data. Inference means applying the fitted model to a new sample. They need not happen on the same computer.

Use a laptop CPU for parsing, plots, data-quality checks and small baselines. If a neural audio model warrants a GPU, use a permitted cloud notebook or a time-limited GPU job. Free Colab offers variable resources, not guaranteed GPU access or uninterrupted service. Do not upload restricted clinical recordings to a personal cloud account without the data-use permission covering that environment. [Colab FAQ](https://research.google.com/colaboratory/intl/en-GB/faq.html)

First run both models on the laptop. Move a small awake model onto the phone after measuring latency and output agreement. Run nighttime inference after recording on the laptop or phone initially. Cloud batch inference can follow when convenient for multiple users. Fly.io or another VM could host an API/job worker later, but no deployment or paid service is needed to test the first model. No credits have been secured in this task, and development should not depend on hypothetical credits. Do not put cloud secrets on the wearable or in the app.

## Existing repository mismatch

A limited code inspection found these specific issues. This was not an audit of the entire repository:

- `device.py:312` defines `BLEStimDevice`, a stimulation-oriented placeholder raising `NotImplementedError`. It is not an implemented ADS1293 recording transport.
- `recording.py:86` describes a finger-worn monitor and requires oxygen/heart-rate data plus an audio envelope.
- `night/features.py` extracts SpO2 features. The present BOM has no SpO2 sensor.
- `night/model.py:516` defines a convolutional detector placeholder that is not implemented.

Create an acquisition-only Arc session schema and BLE recorder. Keep legacy simulations explicitly separate. Do not invent missing oxygen values, present simulated benchmarks as hardware results, or route anything to stimulation methods. This report has not modified those files.

## Sent data outreach already verified

All five initial inquiries below were sent October 19. These are specific checked threads, not an exhaustive count of all historical outreach.

| Contact | Sent request | Current checked status |
|---|---|---|
| Eric Miller, UPMC/Pitt, millerej@upmc.edu | Whether raw sleep audio is retained/aligned to PSG and who authorizes access | Replied November 5: snoring recordings align with PSG, quality depends on setup/filters, access outside his remit. Overview and access-contact follow-up sent November 8. No dataset grant. |
| Sgambati, JHU CISRE, cisre.fsgamba1@jhu.edu | Raw audio with scored PSG, sponsorship and external-student access process | No reply in the checked thread. |
| Portwood, Cleveland Clinic, portwos@ccf.org | Audio content of sleep repository, microphone location, permitted use and sponsorship | No reply in the checked thread. |
| Stelios Potirakis, University of West Attica, spoti@uniwa.gr | PSG-Audio release, synchronization, missing channels and eventual product-use conditions | No reply in the checked thread. |
| Ahmed Elwali, Marian, aelwali@marian.edu | Acquisition suitability and awake-to-night transfer | Offered a meeting. November 4 follow-up sent about reference recordings and external validation. No data-access offer. |

These requests concern recording data or acquisition methods. The patient questionnaire collects needs/preferences and is not training ground truth for either physiological model.

## Next work, in dependency order

1. Aarav resolves the carrier hold and confirms ADC routing/filter configuration. Software team freezes packet and session formats with him.
2. Bench-only capture proves decoding, sample counts, clipping detection, timing and BLE continuity. Progress from minutes to a full-night-duration bench run. Measure runtime and temperature in the intended configuration.
3. Build and test the laptop recorder, then one phone logger. Compare saved samples with the source and record every dropout. Do not infer clinical readiness from a clean bench trace.
4. Obtain one permitted public audio recording with annotations, verify alignment visually and programmatically, then build a subject-split baseline. Request clarification where venture-use permissions are uncertain.
5. Define the awake reference protocol with a qualified clinician and arrange required safety/participant review before body recordings. Gather the specific labelled data that public datasets do not supply.
6. Follow up narrowly on existing data contacts, asking for modality, labels, timing, placement and access conditions. Pursue independent Arc-plus-reference validation only when the recording system works.
7. Compare baselines before adding a larger model. Keep research-only scores separate from patient-facing coaching or treatment decisions.

The highest priority is a trustworthy recording and label pipeline. A model architecture cannot recover a signal that was not captured, repair unmarked packet losses or supply missing clinical truth.
