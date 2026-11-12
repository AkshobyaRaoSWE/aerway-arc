# PSG-Audio: access passes; timing gate remains closed

The current dataset metadata identifies **V3, PUBLIC access, CC BY 4.0**, with no additional restricted-access text. This permits local venture research under the license's attribution, license-link and change-notice requirements, including commercial use. The license does not eliminate privacy rights. This check concerns the dataset license, separately from the paper license. [Dataset](https://www.scidb.cn/en/detail?dataSetId=778740145531650048), [official metadata](https://www.scidb.cn/api/sdb-dataset-service/dataset/details/778740145531650048?version=V3&mode=front), [license](https://creativecommons.org/licenses/by/4.0/).

The V3 README says 287 participants, while the landing-page description still says 212. The downloadable file manifest includes historical V1/V2 paths alongside V3: select explicit `/V3/` paths, not every returned link. Original clinical annotations are under `APNEA_RML`; `APNEA_RML_clean` contains algorithmically filtered annotations only for part of the cohort. Preserve that distinction in any later evaluation. [V3 README](https://china.scidb.cn/download?fileId=6225baffd7561b594fb67e20), [official file manifest](https://www.scidb.cn/api/gin-sdb-filetree/getAllUrl?dataSetId=778740145531650048&type=personal&version=V3&global=Chicago).

## Actual ingestion result

Downloaded the header and **only 60 seconds** of one EDF segment, source record `00001006-100507`, plus its original and cleaned RML annotations. This was an ingestion check, not a complete recording download or accuracy evaluation. The repository contains inspection code and documentation; raw data and generated reports remain gitignored under `local-data/` and `runs/`.

The range is EDF records `[400,460)`, bytes `78975776–90821335`, exactly **11,845,560 bytes**. Both actual audio channels, `Mic` and `Tracheal`, contain 48 kHz samples. Each has 2,880,000 samples in this range and no completely zero one-second records. Header and RML segment timestamps match; two scored obstructive-apnea events overlap the range. The inspector verifies geometry, timestamps, event intervals and audio content. [Source EDF](https://china.scidb.cn/download?fileId=61a0ca3f89f14b48842cf1e0), [original RML](https://china.scidb.cn/download?fileId=61a0ca3f89f14b48842ceed4), [cleaned RML](https://china.scidb.cn/download?fileId=61a0ca3f89f14b48842ced90).

Exact range SHA-256: `0281c71471a0f9972aa110e264f10ebc445c92c9d972d7259ae2f0482205e8a4`.

**Structural ingestion: PASS. Signal alignment: UNVALIDATED. Training allowed: false.** Matching timestamps do not prove audio-to-reference synchronization. The paper reports synchronization tolerances up to two seconds, not the software's existing 50 ms gate. Preliminary envelope correlations were weak and inconsistent across short windows; they do not establish an error or drift bound. Do not enter `0.05` as an invented measurement or silently loosen the gate. [Dataset methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC8333307/).

## Reproduce the bounded check

Run from repository root after installing the pinned environment. Downloads use the official website's public `china.scidb.cn` host. Its generated link list also includes `download.scidb.cn`, which was unreliable during this check. Retain HTTP response headers and verify `206` and the expected `Content-Range` for byte-range downloads.

```sh
mkdir -p local-data/psg-audio/check
curl --fail --location --range 0-65535 --dump-header local-data/psg-audio/check/header.http 'https://china.scidb.cn/download?fileId=61a0ca3f89f14b48842cf1e0' --output local-data/psg-audio/check/header.bin
curl --fail --location --range 78975776-90821335 --dump-header local-data/psg-audio/check/records.http 'https://china.scidb.cn/download?fileId=61a0ca3f89f14b48842cf1e0' --output local-data/psg-audio/check/records.bin
curl --fail --location 'https://china.scidb.cn/download?fileId=61a0ca3f89f14b48842ceed4' --output local-data/psg-audio/check/original.rml
python scripts/inspect_psg_audio.py --header local-data/psg-audio/check/header.bin --records local-data/psg-audio/check/records.bin --rml local-data/psg-audio/check/original.rml --start-record 400 --record-count 60 --source-url 'https://china.scidb.cn/download?fileId=61a0ca3f89f14b48842cf1e0' --output runs/psg-access/ingestion.json
python -m pytest -q tests/test_psg_inspection.py
```

Inspection output includes hashes, exact source-byte interval, actual audio statistics, overlapping events and explicit blocked-training status. Fixture tests verify parsing and rejection behavior; they are not clinical or model-performance evidence. The script deliberately supports one acquisition session with explicit RML segment timing; it rejects unsupported geometry instead of guessing.

## Gate before participant-held-out evaluation

1. Establish and document defensible audio/reference timing uncertainty for the selected V3 recordings, including offset, drift, padding and ambiguous intervals. Obtain authoritative clarification or conduct a robust, reviewed synchronization audit. A correlation peak alone is insufficient.
2. Explicitly approve a night-dataset timing policy suitable for that uncertainty, including exclusion of ambiguous event boundaries. Keep the awake bench calibration/reference gate separate. Preserve original annotation provenance and distinguish automatic filtering.
3. Freeze eligible participant identifiers and separate train/validation/test groups before looking at model results. Define sleep/wake and treatment exclusions from RML and dataset metadata. Use multiple participants and preserve natural event prevalence.
4. Evaluate the existing MFCC/logistic baseline with those fixed groups, training-only preprocessing and validation-only threshold choice. Report participant-level results, exclusions, prevalence and uncertainty. Only then decide whether a larger night model merits work.

No participant-held-out run has been performed. Larger downloads were stopped when the alignment gate remained unresolved. Existing baseline and training gates remain unchanged.
