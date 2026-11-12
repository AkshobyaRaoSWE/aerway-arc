import json

import numpy as np
import pytest

from arc.evaluation import (extract_events, make_splits, match_events,
                            participant_bootstrap, score_predictions, validate_splits)


def rows(labels, probabilities, valid=None, task="night_audio", participant="p1", session="s1"):
    valid = valid if valid is not None else [True] * len(labels)
    return [dict(participant_id=participant, session_id=session, task=task, start_s=float(i), end_s=float(i + 1),
                 label=label, probability=probability, valid=good)
            for i, (label, probability, good) in enumerate(zip(labels, probabilities, valid))]


def test_splits_are_order_independent_and_repeat_sessions_stay_together():
    people = [f"p{i}" for i in range(12)]
    result = make_splits(people)
    assert result == make_splits(list(reversed(people)) + people)
    validate_splits(result, people)
    assert set(result["train"]).isdisjoint(result["test"])
    assert all(result.values())


@pytest.mark.parametrize("splits", [
    {"train": ["a"], "validation": ["a"], "test": ["c"]},
    {"train": ["a"], "validation": ["b"], "test": ["d"]},
    {"train": ["a", "a"], "validation": ["b"], "test": ["c"]},
])
def test_bad_splits_rejected(splits):
    with pytest.raises(ValueError):
        validate_splits(splits, ["a", "b", "c"])


def test_too_few_people_and_bad_fractions_rejected():
    with pytest.raises(ValueError):
        make_splits(["a", "b"])
    with pytest.raises(ValueError):
        make_splits(["a", "b", "c"], validation_fraction=np.nan)


def test_gap_merge_cannot_bridge_invalid_or_absent_time():
    events = extract_events([0, 1, 2], [1, 2, 3], [.8, .1, .9], [True] * 3, .5, gap_merge_s=1)
    assert [(e["start_s"], e["end_s"]) for e in events] == [(0, 3)]
    invalid = extract_events([0, 1, 2], [1, 2, 3], [.8, np.nan, .9], [True, False, True], .5, gap_merge_s=20)
    absent = extract_events([0, 2], [1, 3], [.8, .9], [True, True], .5, gap_merge_s=20)
    assert [(e["start_s"], e["end_s"]) for e in invalid] == [(0, 1), (2, 3)]
    assert invalid == absent


def test_min_duration_after_merge_and_threshold_ties():
    assert len(extract_events([0, 1, 2], [1, 2, 3], [.5, .1, .5], [True] * 3, .5, 3, 1)) == 1
    assert extract_events([0], [1], [.5], [True], .5, 2) == []
    assert extract_events([], [], [], np.array([], dtype=bool), .5) == []


@pytest.mark.parametrize("arguments", [
    ([0, .5], [1, 1.5], [.5, .6], [True, True]),
    ([0], [1], [np.nan], [True]),
    ([0], [1], [1.1], [True]),
    ([0], [1], [.5], [1]),
])
def test_malformed_timeline_rejected(arguments):
    with pytest.raises(ValueError):
        extract_events(*arguments, threshold=.5)


def test_matching_counts_each_reference_once():
    predicted = [{"start_s": 0, "end_s": 4}, {"start_s": 1, "end_s": 5}]
    result = match_events(predicted, [{"start_s": 0, "end_s": 5}])
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 0)


def test_matching_maximizes_cardinality_before_iou():
    predicted = [{"start_s": 0, "end_s": 6}, {"start_s": 0, "end_s": 3}]
    reference = [{"start_s": 0, "end_s": 4}, {"start_s": 4, "end_s": 6}]
    result = match_events(predicted, reference, .3)
    assert result["tp"] == 2
    assert match_events([{"start_s": 0, "end_s": 1}], [{"start_s": 1, "end_s": 2}], 0)["tp"] == 0


def test_censored_reference_not_split_and_predictions_not_false_positive():
    report = score_predictions(rows([0, 1, 1, 1, 0], [.1, .8, None, .9, .1], [True, True, False, True, True]))
    assert report["reference_events"] == 1
    assert report["censored_reference_events"] == 1
    assert report["excluded_predictions"] == 2
    assert report["tp"] == report["fp"] == report["fn"] == 0
    assert report["event_sensitivity"] is None
    assert report["event_evaluable_seconds"] == 2
    assert report["valid_labelled_seconds"] == 4
    assert report["usable_recording_fraction"] == .8


def test_unknown_labels_are_not_negatives():
    report = score_predictions(rows([0, None, 0], [.1, .9, .1]))
    assert report["fp"] == 0
    assert report["valid_labelled_seconds"] == 2
    assert report["average_precision"] is None
    assert report["brier_score"] == pytest.approx(.01)


def test_truth_touching_unknown_is_censored():
    report = score_predictions(rows([None, 1, 0], [.1, .9, .1]))
    assert report["censored_reference_events"] == 1
    assert report["excluded_predictions"] == 1


def test_awake_target_is_interruption_and_metrics_duration_weighted():
    report = score_predictions(rows([1, 0, 0, 1], [.9, .1, .1, .9], task="awake"))
    assert report["event_target"] == "reference-defined interruption"
    assert report["tp"] == 1
    assert report["event_sensitivity"] == 1
    assert report["false_interruptions_per_minute_valid_labelled_time"] == 0
    assert report["coaching_cues_evaluated"] is False
    assert report["brier_score"] == pytest.approx(.01)
    assert report["mean_absolute_onset_error_s"] == 0


def test_empty_all_invalid_and_no_event_denominators_are_null():
    for report in [score_predictions([]), score_predictions(rows([0, 1], [None, None], [False, False]))]:
        assert report["event_sensitivity"] is None
        assert report["false_detections_per_hour"] is None
        assert report["brier_score"] is None
        json.dumps(report, allow_nan=False)


def test_missing_time_counts_against_coverage():
    values = rows([0, 0], [.1, .1])
    values[1].update(start_s=3, end_s=4)
    report = score_predictions(values)
    assert report["total_seconds"] == 4
    assert report["valid_seconds"] == 2
    assert report["usable_recording_fraction"] == .5


def test_bootstrap_is_seeded_and_resamples_people():
    values = rows([0, 1, 0], [.1, .9, .1]) + rows([0, 1, 0], [.1, .1, .1], participant="p2", session="s2")
    report = score_predictions(values)
    interval = report["confidence_intervals"]["intervals"]["event_sensitivity"]
    assert interval["low"] == 0 and interval["high"] == 1
    assert report["confidence_intervals"] == participant_bootstrap(report["per_participant"])
    assert score_predictions(values[:3])["confidence_intervals"]["intervals"] is None


def test_mixed_tasks_or_participant_session_collisions_rejected():
    with pytest.raises(ValueError):
        score_predictions(rows([0], [.1]) + rows([0], [.1], task="awake", session="s2"))
    with pytest.raises(ValueError):
        score_predictions(rows([0], [.1]) + rows([0], [.1], participant="p2"))


def test_empty_input_still_validates_configuration():
    with pytest.raises(ValueError):
        score_predictions([], threshold=float("nan"))
    with pytest.raises(ValueError):
        score_predictions([], iou_threshold=2)


def test_precision_recall_curve_uses_event_target_and_duration():
    report = score_predictions(rows([0, 1, 0], [.1, .9, .2]))
    curve = report["precision_recall_curve"]
    assert curve["thresholds"][-1] is None
    assert curve["precision"][-1] == 1
    assert curve["recall"][-1] == 0
    assert report["average_precision"] == 1


def test_large_sparse_timeline_does_not_need_dense_pair_matrix():
    events = [{"start_s": float(i * 2), "end_s": float(i * 2 + 1)} for i in range(10000)]
    result = match_events(events, events)
    assert result["tp"] == 10000
    assert result["fp"] == result["fn"] == 0


def test_status_distinguishes_no_data_no_reference_and_evaluated():
    assert score_predictions([])["evaluation_status"] == "insufficient_evaluable_data"
    assert score_predictions(rows([0], [.1]))["evaluation_status"] == "no_evaluable_reference_events"
    assert score_predictions(rows([0, 1, 0], [.1, .9, .1]))["evaluation_status"] == "evaluated"


def test_confound_metrics_use_duration_and_overlap_without_fragmented_events():
    values = rows([1, 0, None], [.9, .8, None], [True, True, False])
    values[0].update(confounds=["speech", "movement"], end_s=.25)
    values[1].update(confounds=["speech"], start_s=.25, end_s=1.)
    values[2].update(start_s=1., end_s=2.)
    report = score_predictions(values)
    speech = report["per_confounds"]["speech"]
    assert speech["tp_seconds"] == .25
    assert speech["fp_seconds"] == .75
    assert speech["frame_precision"] == .25
    assert speech["frame_sensitivity"] == 1
    assert report["per_confounds"]["movement"]["annotated_bin_seconds"] == .25
    assert report["per_confounds"]["none_annotated"]["usable_fraction"] == 0
    assert "event_sensitivity" not in speech


def test_awake_confound_metrics_use_interruption_probability():
    values = rows([1, 0], [.9, .9], task="awake")
    values[0]["confounds"] = values[1]["confounds"] = ["swallow"]
    report = score_predictions(values)
    assert report["per_confounds"]["swallow"]["fn_seconds"] == 1
    assert report["per_confounds"]["swallow"]["tn_seconds"] == 1


def test_exact_non_grid_reference_does_not_disappear_at_mixed_label_bins():
    values = rows([0, None, 1, None, 0], [.1, .9, .9, .9, .1])
    for row in values:
        row["scored"] = True
    reference = [{"participant_id": "p1", "session_id": "s1", "start_s": 1.25, "end_s": 3.75}]
    report = score_predictions(values, reference_events=reference)
    assert report["reference_events"] == 1
    assert report["censored_reference_events"] == 0
    assert report["tp"] == 1
    assert report["valid_labelled_seconds"] == 3
    assert report["valid_scored_seconds"] == 5
    assert report["event_evaluable_seconds"] == 5
    assert report["mean_absolute_onset_error_s"] == .25
    assert report["mean_absolute_offset_error_s"] == .25
    assert report["reference_timing"] == "original annotation intervals"


def test_exact_reference_with_invalid_middle_remains_one_censored_event():
    values = rows([0, None, 1, None, 0], [.1, .9, None, .9, .1], [True, True, False, True, True])
    for row in values:
        row["scored"] = True
    reference = [{"participant_id": "p1", "session_id": "s1", "start_s": 1.25, "end_s": 3.75}]
    report = score_predictions(values, reference_events=reference)
    assert report["reference_events"] == report["censored_reference_events"] == 1
    assert report["excluded_predictions"] == 2
    assert report["event_evaluable_seconds"] == 2.5


def test_exact_reference_crossing_missing_bin_or_recording_edge_is_censored():
    values = rows([1, 1], [.9, .9])
    values[1].update(start_s=3., end_s=4.)
    for row in values:
        row["scored"] = True
    for end in (3.5, 4.5):
        reference = [{"participant_id": "p1", "session_id": "s1", "start_s": .5, "end_s": end}]
        report = score_predictions(values, reference_events=reference)
        assert report["censored_reference_events"] == 1


def test_exact_reference_contract_requires_scored_and_correct_session():
    values = rows([0], [.1])
    with pytest.raises(ValueError, match="scored"):
        score_predictions(values, reference_events=[])
    values[0]["scored"] = True
    with pytest.raises(ValueError, match="session"):
        score_predictions(values, reference_events=[{"participant_id": "p1", "session_id": "wrong", "start_s": 0., "end_s": 1.}])
