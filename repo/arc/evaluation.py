"""Participant-isolated splits and research-only timeline evaluation.

Times describe nonoverlapping prediction bins, not the feature-window support.
Unknown labels never become negatives. Censored reference events are not split
into easier events when the device drops data.
"""
from __future__ import annotations

from collections import defaultdict
import heapq
from numbers import Real

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from sklearn.metrics import average_precision_score, precision_recall_curve


def _participants(values):
    values = list(values)
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("participant IDs must be nonempty strings")
    return sorted(set(values))


def make_splits(participant_ids, seed=42, validation_fraction=.2, test_fraction=.2):
    """Save this result before windowing; repeated sessions follow their person."""
    ids = _participants(participant_ids)
    fractions = np.array([validation_fraction, test_fraction], dtype=float)
    if not np.isfinite(fractions).all() or (fractions <= 0).any() or fractions.sum() >= 1:
        raise ValueError("split fractions must be positive and leave training participants")
    if len(ids) < 3:
        raise ValueError("at least three participants required for disjoint splits")
    n_validation = max(1, int(len(ids) * validation_fraction))
    n_test = max(1, int(len(ids) * test_fraction))
    if n_validation + n_test >= len(ids):
        raise ValueError("fractions leave no training participants")
    shuffled = np.random.default_rng(seed).permutation(ids).tolist()
    result = {
        "validation": sorted(shuffled[:n_validation]),
        "test": sorted(shuffled[n_validation:n_validation + n_test]),
        "train": sorted(shuffled[n_validation + n_test:]),
    }
    validate_splits(result, ids)
    return result


def validate_splits(splits, participant_ids):
    if not isinstance(splits, dict) or set(splits) != {"train", "validation", "test"}:
        raise ValueError("exactly train, validation and test splits required")
    seen = set()
    for name, ids in splits.items():
        if not isinstance(ids, list) or not ids or len(_participants(ids)) != len(ids):
            raise ValueError(f"{name}: nonempty unique participant list required")
        if seen.intersection(ids):
            raise ValueError("participant leakage between splits")
        seen.update(ids)
    if seen != set(_participants(participant_ids)):
        raise ValueError("splits must cover exactly the manifest participants")


def _timeline(starts, ends, probabilities, valid):
    starts, ends, probabilities = [np.asarray(x, dtype=float) for x in (starts, ends, probabilities)]
    valid = np.asarray(valid)
    if any(x.ndim != 1 for x in (starts, ends, probabilities, valid)):
        raise ValueError("timeline arrays must be one-dimensional")
    if len({len(x) for x in (starts, ends, probabilities, valid)}) != 1:
        raise ValueError("timeline array lengths differ")
    if valid.dtype.kind != "b":
        raise ValueError("valid mask must be boolean")
    if not np.isfinite(starts).all() or not np.isfinite(ends).all() or (starts < 0).any() or (ends <= starts).any():
        raise ValueError("finite, nonnegative, positive-duration intervals required")
    if len(starts) > 1 and np.any(starts[1:] < ends[:-1] - 1e-9):
        raise ValueError("prediction bins must be sorted and nonoverlapping")
    if not np.isfinite(probabilities[valid]).all() or np.any((probabilities[valid] < 0) | (probabilities[valid] > 1)):
        raise ValueError("valid probabilities must be finite and in [0,1]")
    return starts, ends, probabilities, valid


def extract_events(starts, ends, probabilities, valid, threshold, min_duration_s=0, gap_merge_s=0):
    """Merge only observed valid gaps. Invalid or absent bins always break events."""
    parameters = np.asarray([threshold, min_duration_s, gap_merge_s], dtype=float)
    if not np.isfinite(parameters).all() or not 0 <= threshold <= 1 or min_duration_s < 0 or gap_merge_s < 0:
        raise ValueError("invalid event threshold or duration")
    starts, ends, probabilities, valid = _timeline(starts, ends, probabilities, valid)
    events = []
    first = last = None

    def finish():
        if first is not None and ends[last] - starts[first] >= min_duration_s - 1e-9:
            weights = ends[first:last + 1] - starts[first:last + 1]
            events.append({"start_s": float(starts[first]), "end_s": float(ends[last]),
                           "confidence": float(np.average(probabilities[first:last + 1], weights=weights))})

    for index in range(len(starts)):
        broken = not valid[index] or (index > 0 and starts[index] > ends[index - 1] + 1e-9)
        if broken:
            finish()
            first = last = None
        if not valid[index]:
            continue
        if probabilities[index] >= threshold:
            if first is not None and starts[index] - ends[last] > gap_merge_s + 1e-9:
                finish()
                first = last = None
            if first is None:
                first = index
            last = index
    finish()
    return events


def _intervals(events):
    intervals = np.asarray([[event["start_s"], event["end_s"]] for event in events], dtype=float).reshape(-1, 2)
    if not np.isfinite(intervals).all() or (intervals[:, 0] < 0).any() or (intervals[:, 1] <= intervals[:, 0]).any():
        raise ValueError("events require finite positive-duration intervals")
    return intervals


def match_events(predicted, reference, iou_threshold=.3):
    """Maximum-cardinality one-to-one matching; total IoU breaks count ties."""
    if not isinstance(iou_threshold, Real) or not np.isfinite(iou_threshold) or not 0 <= iou_threshold <= 1:
        raise ValueError("IoU threshold must be in [0,1]")
    p, r = _intervals(predicted), _intervals(reference)
    matches = []
    if len(p) and len(r):
        # Sparse overlaps keep noisy full-night timelines from allocating P x R.
        points = sorted([(start, 0, i) for i, (start, _) in enumerate(p)] +
                        [(start, 1, i) for i, (start, _) in enumerate(r)])
        active, expirations, overlaps = [{}, {}], [], {}
        for start, side, index in points:
            while expirations and expirations[0][0] <= start:
                _, expired_side, expired_index = heapq.heappop(expirations)
                del active[expired_side][expired_index]
            interval = (p, r)[side][index]
            for other_index, other in active[1 - side].items():
                intersection = min(interval[1], other[1]) - start
                union = interval[1] - interval[0] + other[1] - other[0] - intersection
                iou = intersection / union
                if iou >= iou_threshold:
                    pair = (index, other_index) if side == 0 else (other_index, index)
                    overlaps[pair] = float(iou)
            active[side][index] = interval
            heapq.heappush(expirations, (interval[1], side, index))
        # Each prediction has its own unmatched column so a full matching exists.
        pi = list(range(len(p)))
        ri = list(range(len(r), len(r) + len(p)))
        weights = [1.] * len(p)
        cardinality_weight = min(len(p), len(r)) + 1
        for (prediction_index, reference_index), iou in overlaps.items():
            pi.append(prediction_index)
            ri.append(reference_index)
            weights.append(-(cardinality_weight + iou))
        graph = coo_matrix((weights, (pi, ri)), shape=(len(p), len(r) + len(p))).tocsr()
        prediction_indices, reference_indices = min_weight_full_bipartite_matching(graph)
        for pi, ri in zip(prediction_indices, reference_indices):
            if ri < len(r):
                matches.append({"prediction_index": int(pi), "reference_index": int(ri),
                                "iou": overlaps[pi, ri],
                                "onset_error_s": float(p[pi, 0] - r[ri, 0]),
                                "offset_error_s": float(p[pi, 1] - r[ri, 1])})
    return {"matches": matches, "tp": len(matches), "fp": len(p) - len(matches), "fn": len(r) - len(matches)}


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _rates(counts):
    return {
        "event_sensitivity": _ratio(counts["tp"], counts["tp"] + counts["fn"]),
        "event_precision": _ratio(counts["tp"], counts["tp"] + counts["fp"]),
        "false_detections_per_hour": _ratio(3600 * counts["fp"], counts["event_evaluable_seconds"]),
        "usable_recording_fraction": _ratio(counts["valid_seconds"], counts["total_seconds"]),
        "labelled_coverage_fraction": _ratio(counts["labelled_seconds"], counts["total_seconds"]),
    }


def participant_bootstrap(participant_summaries, seed=42, repetitions=1000):
    """Aggregate counts after resampling people, never correlated windows."""
    if not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    records = list(participant_summaries)
    if len(records) < 2:
        return {"method": "participant percentile bootstrap", "participants": len(records),
                "repetitions": 0, "intervals": None, "reason": "fewer than two participants"}
    keys = ("tp", "fp", "fn", "event_evaluable_seconds", "valid_seconds", "total_seconds", "labelled_seconds")
    matrix = np.asarray([[row[key] for key in keys] for row in records], dtype=float)
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("participant summaries must have finite nonnegative counts")
    metrics = defaultdict(list)
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        sampled = matrix[rng.integers(0, len(records), len(records))].sum(axis=0)
        for name, value in _rates(dict(zip(keys, sampled))).items():
            if value is not None:
                metrics[name].append(value)
    intervals = {}
    for name in _rates(dict(zip(keys, matrix.sum(axis=0)))):
        values = metrics[name]
        intervals[name] = ({"low": float(np.percentile(values, 2.5)), "high": float(np.percentile(values, 97.5)),
                            "defined_replicates": len(values)} if values else None)
    return {"method": "participant percentile bootstrap", "participants": len(records),
            "repetitions": repetitions, "seed": seed, "intervals": intervals}


def score_predictions(rows, threshold=.5, min_duration_s=0, gap_merge_s=0, iou_threshold=.3, reference_events=None):
    """Score one task; awake event metrics concern interruption, not maintained hold.

    Primary event sensitivity excludes any reference event partly invalid, or
    touching unknown/missing-time bins. Predictions touching those censored
    events are excluded, not labelled false. Their full spans are removed from
    event exposure. Reported frame metrics retain all valid labelled bins.
    Exact reference_events override binned truth; each row then requires scored,
    meaning its entire bin has independent positive/negative annotation coverage.
    Mixed-state boundary bins remain usable for event scoring, not frame scoring.
    """
    extract_events([], [], [], np.array([], dtype=bool), threshold, min_duration_s, gap_merge_s)
    match_events([], [], iou_threshold)
    rows = list(rows)
    groups = defaultdict(list)
    tasks = set()
    owners = {}
    for row in rows:
        for field in ("participant_id", "session_id"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"{field} required")
        participant, session = row["participant_id"], row["session_id"]
        if session in owners and owners[session] != participant:
            raise ValueError("session assigned to multiple participants")
        owners[session] = participant
        if row.get("task") not in ("awake", "night_audio"):
            raise ValueError("task must be awake or night_audio")
        tasks.add(row["task"])
        if type(row.get("valid")) is not bool:
            raise ValueError("valid must be boolean")
        label = row.get("label")
        if label is not None and (not isinstance(label, Real) or label not in (0, 1)):
            raise ValueError("label must be 0, 1 or null")
        if row["valid"] and (not isinstance(row.get("probability"), Real) or not np.isfinite(row["probability"])):
            raise ValueError("valid prediction requires a finite probability")
        groups[(participant, session)].append(row)
    if len(tasks) > 1:
        raise ValueError("evaluate awake and night_audio separately")
    exact_references = None
    if reference_events is not None:
        exact_references = defaultdict(list)
        for event in reference_events:
            key = (event.get("participant_id"), event.get("session_id"))
            if key not in groups:
                raise ValueError("reference event has no matching prediction session")
            _intervals([event])
            exact_references[key].append(event)
        for events in exact_references.values():
            events.sort(key=lambda event: event["start_s"])
            if any(current["start_s"] < previous["end_s"] for previous, current in zip(events, events[1:])):
                raise ValueError("reference events cannot overlap within a session")
        if any(type(row.get("scored")) is not bool for row in rows):
            raise ValueError("exact-reference evaluation requires explicit scored flags")
    task = next(iter(tasks), None)
    summaries = defaultdict(lambda: defaultdict(float))
    confound_counts = defaultdict(lambda: defaultdict(float))
    all_labels, all_probabilities, all_weights, all_matches = [], [], [], []
    keys = ("tp", "fp", "fn", "total_seconds", "valid_seconds", "labelled_seconds", "valid_labelled_seconds",
            "scored_seconds", "valid_scored_seconds", "event_evaluable_seconds", "reference_events", "censored_reference_events", "predicted_events", "excluded_predictions")
    for (participant, session), group in sorted(groups.items()):
        group.sort(key=lambda row: row["start_s"])
        starts = np.array([row["start_s"] for row in group], dtype=float)
        ends = np.array([row["end_s"] for row in group], dtype=float)
        probabilities = np.array([row["probability"] if row["valid"] else np.nan for row in group], dtype=float)
        valid = np.array([row["valid"] for row in group], dtype=bool)
        starts, ends, probabilities, valid = _timeline(starts, ends, probabilities, valid)
        known = np.array([row.get("label") is not None for row in group], dtype=bool)
        labels = np.array([row["label"] if row.get("label") is not None else np.nan for row in group], dtype=float)
        if task == "awake":
            probabilities, labels = 1 - probabilities, 1 - labels
        durations = ends - starts
        evaluable = valid & known
        for index, row in enumerate(group):
            confounds = row.get("confounds", [])
            if not isinstance(confounds, list) or any(not isinstance(c, str) or not c.strip() for c in confounds):
                raise ValueError("confounds must be a list of nonempty strings")
            for confound in set(confounds) or {"none_annotated"}:
                counts = confound_counts[confound]
                duration = float(durations[index])
                counts["annotated_bin_seconds"] += duration
                counts["valid_seconds"] += duration * bool(valid[index])
                counts["labelled_seconds"] += duration * bool(known[index])
                if evaluable[index]:
                    counts["valid_labelled_seconds"] += duration
                    positive = probabilities[index] >= threshold
                    correct = labels[index] == 1
                    name = "tp" if positive and correct else "fp" if positive else "fn" if correct else "tn"
                    counts[name + "_seconds"] += duration
        scored = known if exact_references is None else np.array([row["scored"] for row in group], dtype=bool)
        event_eligible = valid & scored
        truth = (extract_events(starts, ends, labels, known, .5) if exact_references is None
                 else exact_references[(participant, session)])
        predictions = extract_events(starts, ends, probabilities, event_eligible, threshold, min_duration_s, gap_merge_s)
        censored, references = [], []
        event_exposure = durations * event_eligible
        for event in truth:
            left = int(np.searchsorted(ends, event["start_s"], side="right"))
            right = int(np.searchsorted(starts, event["end_s"], side="left")) - 1
            indices = slice(left, right + 1)
            overlap = np.maximum(0, np.minimum(ends[indices], event["end_s"]) - np.maximum(starts[indices], event["start_s"]))
            observed = float(overlap[event_eligible[indices]].sum())
            is_censored = observed < event["end_s"] - event["start_s"] - 1e-9
            if exact_references is None:
                unknown_left = left > 0 and (not known[left - 1] or starts[left] > ends[left - 1] + 1e-9)
                unknown_right = right + 1 < len(starts) and (not known[right + 1] or starts[right + 1] > ends[right] + 1e-9)
                is_censored = is_censored or unknown_left or unknown_right
            if is_censored:
                censored.append(event)
                event_exposure[indices] -= overlap * event_eligible[indices]
            else:
                references.append(event)
        censored_ends = [event["end_s"] for event in censored]
        eligible_predictions = []
        for event in predictions:
            index = int(np.searchsorted(censored_ends, event["start_s"], side="right"))
            if index == len(censored) or censored[index]["start_s"] >= event["end_s"]:
                eligible_predictions.append(event)
        matched = match_events(eligible_predictions, references, iou_threshold)
        all_matches.extend(matched["matches"])
        counts = summaries[participant]
        counts["sessions"] += 1
        for key in ("tp", "fp", "fn"):
            counts[key] += matched[key]
        # Missing bins inside the represented session span count as unavailable time.
        counts["total_seconds"] += float(ends[-1] - starts[0])
        counts["valid_seconds"] += float(durations[valid].sum())
        counts["labelled_seconds"] += float(durations[known].sum())
        counts["valid_labelled_seconds"] += float(durations[evaluable].sum())
        counts["scored_seconds"] += float(durations[scored].sum())
        counts["valid_scored_seconds"] += float(durations[event_eligible].sum())
        counts["event_evaluable_seconds"] += float(np.maximum(event_exposure, 0).sum())
        counts["reference_events"] += len(truth)
        counts["censored_reference_events"] += len(censored)
        counts["predicted_events"] += len(predictions)
        counts["excluded_predictions"] += len(predictions) - len(eligible_predictions)
        all_labels.extend(labels[evaluable].tolist())
        all_probabilities.extend(probabilities[evaluable].tolist())
        all_weights.extend(durations[evaluable].tolist())
    participants = []
    for participant, counts in sorted(summaries.items()):
        record = {"participant_id": participant, **{key: counts[key] for key in keys}, "sessions": int(counts["sessions"])}
        record.update(_rates(record))
        participants.append(record)
    totals = {key: sum(row[key] for row in participants) for key in keys}
    result = {"task": task, "event_target": "reference-defined interruption" if task == "awake" else "respiratory event candidate",
              "threshold_on_event_probability": threshold, "min_duration_s": min_duration_s, "gap_merge_s": gap_merge_s,
              "iou_threshold": iou_threshold, "participants": len(participants), "sessions": len(groups),
              **totals, **_rates(totals), "per_participant": participants,
              "event_exposure_definition": "valid independently labelled time excluding censored reference-event spans",
              "reference_censoring": ("any invalid, unscored or missing portion of exact reference interval" if exact_references is not None
                                      else "any invalid portion or adjacent unknown/missing-time bin; not split across dropout"),
              "reference_timing": "original annotation intervals" if exact_references is not None else "prediction-bin approximation",
              "average_precision": None, "precision_recall_curve": None, "brier_score": None,
              "mean_absolute_onset_error_s": None, "mean_absolute_offset_error_s": None, "mean_matched_iou": None}
    if all_weights:
        y, p, w = np.asarray(all_labels), np.asarray(all_probabilities), np.asarray(all_weights)
        result["brier_score"] = float(np.average((p - y) ** 2, weights=w))
        if np.any(y == 1):
            result["average_precision"] = float(average_precision_score(y, p, sample_weight=w))
            precision, recall, thresholds = precision_recall_curve(y, p, sample_weight=w)
            indices = np.unique(np.linspace(0, len(precision) - 1, min(101, len(precision)), dtype=int))
            result["precision_recall_curve"] = {
                "precision": precision[indices].tolist(), "recall": recall[indices].tolist(),
                "thresholds": [float(thresholds[i]) if i < len(thresholds) else None for i in indices],
                "points_downsampled": len(indices) < len(precision),
                "weighting": "prediction-bin duration",
            }
    if all_matches:
        for field in ("onset_error_s", "offset_error_s"):
            result["mean_absolute_" + field] = float(np.mean([abs(match[field]) for match in all_matches]))
        result["mean_matched_iou"] = float(np.mean([match["iou"] for match in all_matches]))
    if task == "awake":
        result["false_interruptions_per_minute_valid_labelled_time"] = _ratio(60 * totals["fp"], totals["event_evaluable_seconds"])
        result["coaching_cues_evaluated"] = False
        result["instructed_practice_rate_available"] = False
    result["evaluation_status"] = (
        "insufficient_evaluable_data" if totals["valid_labelled_seconds"] == 0 and totals["event_evaluable_seconds"] == 0 else
        "no_evaluable_reference_events" if totals["tp"] + totals["fn"] == 0 else "evaluated")
    result["per_confounds"] = {}
    for confound, counts in sorted(confound_counts.items()):
        keys = ("annotated_bin_seconds", "valid_seconds", "labelled_seconds", "valid_labelled_seconds",
                "tp_seconds", "fp_seconds", "fn_seconds", "tn_seconds")
        record = {key: counts[key] for key in keys}
        record.update(
            frame_sensitivity=_ratio(counts["tp_seconds"], counts["tp_seconds"] + counts["fn_seconds"]),
            frame_specificity=_ratio(counts["tn_seconds"], counts["tn_seconds"] + counts["fp_seconds"]),
            frame_precision=_ratio(counts["tp_seconds"], counts["tp_seconds"] + counts["fp_seconds"]),
            usable_fraction=_ratio(counts["valid_seconds"], counts["annotated_bin_seconds"]),
            evaluable_fraction=_ratio(counts["valid_labelled_seconds"], counts["annotated_bin_seconds"]))
        result["per_confounds"][confound] = record
    result["confound_summary_note"] = (
        "Duration-weighted frame metrics on event probability; overlapping annotations count in each bucket. "
        "none_annotated means no confound annotation, not proven absence. No fragmented event metrics.")
    result["confidence_intervals"] = participant_bootstrap(participants)
    return result
