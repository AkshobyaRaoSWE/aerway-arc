# Aerway Arc: Akshobya's two-model build brief

8 November 2026. Proposed implementation specification, not a claim that either model is trained or validated. Read alongside Arc_Data_and_ML_Plan.md for hardware, transport, dataset sources and correspondence.

## Build these two tasks

| | Daytime model | Nighttime model |
|---|---|---|
| First question | Can Arc identify maintenance versus interruption of a precisely defined, independently measured tongue hold? | Can microphone audio identify intervals containing independently scored apnea/hypopnea events? |
| Initial inputs | Regional submental/jaw EMG channels, device motion, recording-quality flags | Microphone waveform. Motion can be tested later as an extra input. |
| Independent labels | Synchronized clinician-approved pressure/contact reference plus task/confound annotations | Synchronized clinical PSG or suitable polygraph event annotations |
| First output | Probability of maintained reference-defined hold, with an uncertainty/invalid state | Per-time-step respiratory-event probability, candidate intervals and invalid-data mask |
| Useful summaries | Attempt duration, maintained-hold time, interruptions and usable recording fraction | Candidate event timeline, confidence and usable audio duration for research review |
| Cannot claim yet | Complete exercise correctness, isolated genioglossus activity, tongue position or OSA benefit | Diagnosis, reliable apnea subtype, clinical AHI, treatment success or safe changes to care |

The models share a recorder and quality logging. They do not share labels. Questionnaire answers are patient-needs evidence, not sensor ground truth. Overnight estimates must not automatically choose next-day treatment or exercises.

## 1. Daytime: a calibrated hold/interruption classifier

### Target definition before training

Use one clinician-defined task and a reference criterion. A possible first task is a sustained tongue-to-palate hold. Ask the clinician to define target pressure/contact, required duration and what counts as a meaningful interruption. Do not choose those clinical cutoffs just because they make the classifier easier.

The reference labels indicate whether that defined hold is maintained. They do not certify all aspects of tongue placement or therapeutic technique. Record swallowing, speech, jaw clench and head movement separately. The instruction “hold now” is an event marker, not proof that the person actually held the target.

### Proposed first model

Use a small class-balanced random forest as the initial supervised baseline, alongside a simple calibrated amplitude/timing threshold. Starting experimental settings could be 200 trees and bounded depth, tuned only inside training/validation data. These are starting choices, not validated optimal settings.

Begin with 0.5-second trailing windows updated every 0.1 seconds. Candidate features: RMS, mean absolute amplitude, waveform length, calibrated relative amplitude, short-term slope, valid channel ratios and accelerometer/gyro movement magnitude. Compare 0.25-, 0.5- and 1-second windows within validation because stability and feedback latency trade off. Do not use future samples or zero-phase offline filtering to claim live latency.

Keep raw data. Choose preprocessing only after Aarav confirms ADC bandwidth, sample rate and scaling. Avoid applying a nominal 20–450 Hz bandpass to a recording whose effective bandwidth is only about 325 Hz. Validate any mains notch and noise reduction against the actual captured signal. Normalize using a documented calibration available at deployment, not statistics from a person's future labelled trials.

### Quality and output behavior

Use a separate deterministic validity gate for missing samples, clipping, disconnected leads where detected, and unusable motion/contact. An invalid window means “cannot assess,” not “failed exercise.” The classifier then distinguishes maintained versus non-maintained reference-defined hold in valid windows. Include confound-only intervals as negatives and retain overlapping event annotations for analysis.

A pause/retry cue should require a validated probability threshold and sustained evidence, with a cooldown to avoid repeated cues. Initially save predictions and compare them offline before enabling live coaching. Latency includes the feature window, buffering, transport, processing and the confirmation interval. Do not advertise a 100 ms response simply because windows update at 10 Hz.

### Training table and split

Each row/window must have participant ID, session ID, placement/reattachment ID, task, sample/time bounds, raw-file references, calibration source, independent label source, confound labels and validity. Keep all windows from a person together for the primary unseen-person test. Also assess a clearly separate personalized-calibration scenario with later sessions held out. A random split of adjacent windows is unacceptable.

Report sensitivity to reference-defined interruptions, false cues per minute of valid instructed practice, event timing error, and fraction of time the system abstains. Also report performance for swallowing, jaw clenching, speech and motion separately. Confidence intervals should resample participants, not treat thousands of adjacent windows as independent patients.

Only test audio as an added daytime feature after the EMG/motion baseline exists. Run EMG-only versus EMG-plus-motion versus EMG-plus-motion-plus-audio comparisons on the same held-out subjects. A silent exercise may give the microphone little useful task information.

## 2. Nighttime: audio respiratory-event segmentation

### Target and labels

For the first experiment, combine independently scored apnea and hypopnea intervals into a binary respiratory-event target. Preserve the original clinical subtype labels in the annotation file so errors can be inspected by type. Call the resulting output an apnea/hypopnea event candidate, not an obstructive-apnea diagnosis.

The negative class is valid audio without an annotated target event. Wake, missing/unscored intervals, speech, equipment noise and CPAP titration need explicit handling. Do not casually label every interval outside an event “normal breathing.” Keep diagnostic and treatment portions distinct.

### Proposed first model and comparison

First establish an energy/MFCC or frozen audio-embedding baseline with a simple classifier. The proposed temporal model after that is a small convolutional recurrent network: log-mel spectrogram -> compact convolution layers -> bidirectional GRU -> per-time-step probability head. It is an offline morning-analysis model, so bidirectional context is allowed. An LSTM is another experiment, not a requirement imposed by time-series data.

Provisional frontend: mono 16 kHz PCM, 32 ms STFT frames, 10 ms hop, 64 mel bins, 20-second context. Evaluate the acoustic bandwidth and preprocessing choice rather than treating these defaults as fixed. Produce probabilities at a modest documented temporal resolution, such as 100 ms, and evaluate against the uncertainty of the reference annotations. Model resolution is not proof of that timing accuracy.

Train with class-weighted binary cross entropy initially. Compare imbalance approaches on validation data, not the held-out test set. At evaluation, retain natural event prevalence. For overlapping inference blocks, stitch probabilities on the original timeline before event extraction. Otherwise the same event can be counted multiple times.

Choose probability thresholds, minimum event durations and gap-merging rules on validation data with clinical input. Freeze them before testing. Do not invent a clinical event label from “ten seconds of silence.” A detached microphone, quiet breathing and a partner's snoring are necessary failure cases.

### Public-data starting point

PSG-Audio is the main candidate because it pairs breathing audio with scored sleep recordings. Verify the exact release, permission terms and alignment corrections, then inspect one recording before committing to the full download. A literature description is not proof that the current files download correctly.

APSAA is another useful audio-plus-reference source for possible external evaluation. Its official record says academic/non-commercial use only and no redistribution without permission. Aerway is an independent venture, so obtain clarification before incorporating it into this project. Do not assume competition participation automatically makes all use academic.

YAMNet supplies general audio embeddings, not apnea labels. ApneaWhisper is a relevant published architecture, but no ready-to-run task-specific checkpoint was verified in this review. Do not describe importing a generic pretrained encoder as obtaining a validated apnea detector.

### Evaluation

Split by participant before windowing, preserve repeated nights together, and reserve an untouched test set. If enough permitted data are available, use a second dataset or institution for external testing after the entire pipeline is frozen. A random second split of the same clinic is not the same as external validation.

Report event sensitivity, precision, false detections per hour, precision-recall curves, timing overlap and usable-recording coverage. Inspect performance by microphone configuration, position, noise and treatment status when metadata permit. Include participant-level uncertainty.

Do not call events per recording hour AHI. Without independently justified sleep duration, label the denominator honestly. Model agreement with a public dataset also does not establish performance on Arc's under-chin microphone. Eventually collect appropriately approved Arc audio simultaneously with a clinical reference for the actual hardware transfer test.

## 3. Shared software deliverables

1. A session reader that preserves raw sensor data, configuration and timing, with explicit gap masks.
2. A dataset manifest with source/version, license or permission, participant/session IDs, channel availability and label provenance.
3. Reproducible subject-level split files. Save the IDs before training.
4. Separate preprocessing modules for EMG and audio, with parameters saved beside each model.
5. Baseline-training scripts, evaluation scripts and reports that can be rerun from fixed manifests.
6. Model bundles containing weights, normalization, feature order, units, thresholds, intended task, required channels and software version.
7. An inference function returning probabilities, validity and model version. Missing required channels should cause a documented failure or abstention.
8. A small labelled replay set for checking that laptop and phone predictions agree after export. Only retain/share recordings as permitted.

Suggested layout: `acquisition/`, `data/manifests/`, `labels/`, `splits/`, `models/awake/`, `models/night_audio/`, `evaluation/`, `reports/`. Keep synthetic data in a clearly named separate folder.

A minimal prediction record could contain: session ID, model version, stream-time start/end, probability, valid/invalid status, reason, calibration ID and estimated processing latency. Never serialize an unvalidated result as “OSA improved.”

## 4. Runtime and hardware handoff

Start with laptop inference. Later move the small daytime model to the phone for offline feedback. Nighttime inference can run after the session on a laptop or phone. A VM can be added for batch work later. Training may use a GPU notebook if needed and allowed by the data terms. Continuous cloud streaming is not required.

Aarav must supply an exact channel map and ADC configuration, captured test files, timing/sequence behavior and sustained BLE measurements. Akshobya and Sathvik should implement the receiver and file format together with him. A 24-bit register does not guarantee a clean or correctly scaled EMG waveform.

The current repository does not yet match this specification. `device.py` has an unimplemented stimulation-oriented BLE class. `recording.py` and nighttime feature extraction expect SpO2/heart rate from a different monitor. The present BOM has neither an oximeter nor a stimulator. Do not fill absent inputs with plausible constants. Implement acquisition-only interfaces for this hardware and isolate legacy simulations.

## First milestone

A saved session that another teammate can load, plot and replay with verified timestamps, units, channel identity and missing-data flags. Then a reproducible model baseline with independent labels and honest held-out evaluation. Neither large-scale training nor renting a VM should precede those checks.

Sources and dataset links: see the companion Arc_Data_and_ML_Plan.md. No models were trained and no participant data were collected while preparing these briefs.
