"""Small supervised baselines exported as inspectable JSON, never pickle."""

import numpy as np
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import __version__


def _matrix(X):
    x = np.asarray(X, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError("features must be a finite two-dimensional matrix")
    return x


def fit_baseline(X, y, task, feature_names, seed=42):
    """Fit on training participants only. Caller owns split and provenance checks."""
    x = _matrix(X)
    labels = np.asarray(y)
    if labels.shape != (len(x),) or set(labels.tolist()) != {0, 1}:
        raise ValueError("training requires aligned binary labels and both classes")
    if len(feature_names) != x.shape[1] or len(set(feature_names)) != len(feature_names) or any(
        not isinstance(name, str) or not name.strip() for name in feature_names
    ):
        raise ValueError("feature names must be unique, nonempty, and match X")
    bundle = {"schema_version": 1, "software_version": __version__, "task": task,
              "feature_names": list(feature_names), "probability_calibration": "uncalibrated",
              "seed": int(seed), "training_rows": len(x)}
    if task == "awake":
        model = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=2,
                                       class_weight="balanced", random_state=seed, n_jobs=1)
        model.fit(x, labels)
        trees = []
        for estimator in model.estimators_:
            tree = estimator.tree_
            counts = tree.value[:, 0, :]
            positive = counts[:, 1] / counts.sum(axis=1)
            trees.append({"left": tree.children_left.tolist(), "right": tree.children_right.tolist(),
                          "feature": tree.feature.tolist(), "threshold": tree.threshold.tolist(),
                          "probability": positive.tolist()})
        bundle.update(algorithm="random_forest", trees=trees,
                      parameters={"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 2,
                                  "class_weight": "balanced"})
    elif task == "night_audio":
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
        model.fit(scaler.transform(x), labels)
        bundle.update(algorithm="logistic_regression", mean=scaler.mean_.tolist(),
                      scale=scaler.scale_.tolist(), coefficient=model.coef_[0].tolist(),
                      intercept=float(model.intercept_[0]),
                      parameters={"class_weight": "balanced", "max_iter": 1000})
    else:
        raise ValueError("task must be awake or night_audio")
    validate_bundle(bundle)
    return bundle


def fit_awake_threshold(X, y, feature_names):
    """Training-only balanced-accuracy comparator on mean calibrated RMS."""
    x = _matrix(X)
    labels = np.asarray(y)
    if labels.shape != (len(x),) or set(labels.tolist()) != {0, 1}:
        raise ValueError("training requires aligned binary labels and both classes")
    if len(feature_names) != x.shape[1] or any(not isinstance(n, str) for n in feature_names):
        raise ValueError("feature contract differs from X")
    indices = [i for i, name in enumerate(feature_names) if name.endswith(".relative_rms")]
    if not indices:
        raise ValueError("threshold comparator requires relative_rms features")
    scores = x[:, indices].mean(axis=1)
    order = np.argsort(scores, kind="stable")
    sorted_scores, sorted_labels = scores[order], labels[order]
    candidates = np.r_[np.unique(scores), np.nextafter(scores.max(), np.inf)]
    if not np.isfinite(candidates).all():
        raise ValueError("threshold scores exceed numeric range")
    cut = np.searchsorted(sorted_scores, candidates, side="left")
    positives_before = np.r_[0, np.cumsum(sorted_labels)][cut]
    positives = labels.sum()
    negatives = len(labels) - positives
    balanced = ((positives - positives_before) / positives
                + (cut - positives_before) / negatives) / 2
    bundle = {"schema_version": 1, "software_version": __version__, "task": "awake",
              "feature_names": list(feature_names), "algorithm": "amplitude_threshold",
              "probability_calibration": "hard_threshold_not_probability",
              "indices": indices, "threshold": float(candidates[np.argmax(balanced)]),
              "training_rows": len(x), "parameters": {"selection": "training_balanced_accuracy"}}
    validate_bundle(bundle)
    return bundle


def _vector(value, length, name):
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {name}") from exc
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"invalid {name}")
    return array


def validate_bundle(bundle):
    if not isinstance(bundle, dict) or bundle.get("schema_version") != 1:
        raise ValueError("unsupported model bundle schema")
    names = bundle.get("feature_names")
    if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n.strip() for n in names):
        raise ValueError("missing feature contract")
    if len(set(names)) != len(names):
        raise ValueError("duplicate feature names")
    dimension = len(names)
    algorithm = bundle.get("algorithm")
    if algorithm == "amplitude_threshold" and bundle.get("task") == "awake":
        indices = bundle.get("indices")
        if not isinstance(indices, list) or not indices or any(
            type(i) is not int or i < 0 or i >= dimension for i in indices
        ) or len(set(indices)) != len(indices):
            raise ValueError("invalid amplitude feature indices")
        if any(not names[i].endswith(".relative_rms") for i in indices):
            raise ValueError("threshold requires relative RMS features")
        threshold = bundle.get("threshold")
        if not isinstance(threshold, (float, int)) or not np.isfinite(threshold):
            raise ValueError("invalid amplitude threshold")
    elif algorithm == "logistic_regression" and bundle.get("task") == "night_audio":
        _vector(bundle.get("mean"), dimension, "mean")
        if np.any(_vector(bundle.get("scale"), dimension, "scale") <= 0):
            raise ValueError("scale must be positive")
        _vector(bundle.get("coefficient"), dimension, "coefficient")
        intercept = bundle.get("intercept")
        if not isinstance(intercept, (float, int)) or not np.isfinite(intercept):
            raise ValueError("invalid intercept")
    elif algorithm == "random_forest" and bundle.get("task") == "awake":
        trees = bundle.get("trees")
        if not isinstance(trees, list) or not 1 <= len(trees) <= 1000:
            raise ValueError("invalid forest")
        for tree in trees:
            if not isinstance(tree, dict) or not isinstance(tree.get("left"), list):
                raise ValueError("invalid tree")
            size = len(tree["left"])
            if not 1 <= size <= 100000:
                raise ValueError("invalid tree size")
            left, right, feature = (_vector(tree.get(key), size, key) for key in ("left", "right", "feature"))
            if any(np.any(a != np.floor(a)) for a in (left, right, feature)):
                raise ValueError("tree indices must be integers")
            _vector(tree.get("threshold"), size, "threshold")
            probability = _vector(tree.get("probability"), size, "probability")
            if np.any((probability < 0) | (probability > 1)):
                raise ValueError("tree probabilities outside [0, 1]")
            leaves = (left == -1) & (right == -1)
            nodes = np.flatnonzero(~leaves)
            if np.any((left[nodes] <= nodes) | (left[nodes] >= size)
                      | (right[nodes] <= nodes) | (right[nodes] >= size)):
                raise ValueError("tree has invalid children or a cycle")
            if np.any((feature[nodes] < 0) | (feature[nodes] >= dimension)):
                raise ValueError("tree feature index outside input dimensions")
    else:
        raise ValueError("unsupported algorithm/task pair")
    return bundle


def predict_proba(bundle, X):
    """Return class-1 probabilities, not clinical confidence or a calibrated risk."""
    validate_bundle(bundle)
    x = _matrix(X)
    if x.shape[1] != len(bundle["feature_names"]):
        raise ValueError("feature count differs from model contract")
    if bundle["algorithm"] == "amplitude_threshold":
        return (x[:, bundle["indices"]].mean(axis=1) >= bundle["threshold"]).astype(float)
    if bundle["algorithm"] == "logistic_regression":
        with np.errstate(over="raise", invalid="raise"):
            try:
                logits = ((x - bundle["mean"]) / bundle["scale"]) @ np.asarray(bundle["coefficient"]) + bundle["intercept"]
            except FloatingPointError as exc:
                raise ValueError("features overflow model scaling") from exc
        return expit(logits)
    # sklearn forests compare float32 inputs even though exported thresholds are float64.
    with np.errstate(over="ignore"):
        x32 = x.astype(np.float32)
    if not np.isfinite(x32).all():
        raise ValueError("features exceed forest numeric range")
    output = np.zeros(len(x))
    for tree in bundle["trees"]:
        left, right, feature = (np.asarray(tree[key], dtype=int) for key in ("left", "right", "feature"))
        thresholds = np.asarray(tree["threshold"])
        node = np.zeros(len(x), dtype=int)
        active = np.flatnonzero(left[node] != -1)
        while len(active):
            current = node[active]
            take_left = x32[active, feature[current]] <= thresholds[current]
            node[active] = np.where(take_left, left[current], right[current])
            active = active[left[node[active]] != -1]
        output += np.asarray(tree["probability"])[node]
    return output / len(bundle["trees"])
