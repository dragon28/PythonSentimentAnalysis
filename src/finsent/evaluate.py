"""Score every model on the same held-out tweets and write results/."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from finsent import LABELS, data, models

SEED = 42


def bootstrap_ci(y, p, n=1000, seed=SEED):
    """95% percentile bootstrap interval for macro-F1 (resampling test tweets)."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    s = [f1_score(y[i], p[i], average="macro", labels=LABELS, zero_division=0) for i in idx]
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def score(y, p) -> dict:
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=LABELS, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", labels=LABELS, zero_division=0)),
        "per_class": {
            c: {"precision": float(a), "recall": float(b), "f1": float(f), "support": int(s)}
            for c, a, b, f, s in zip(LABELS, pr, rc, f1, sup, strict=True)
        },
        "confusion": confusion_matrix(y, p, labels=LABELS).tolist(),
    }


def tune_tfidf(texts, labels) -> tuple[float, float]:
    """Pick C by 5-fold CV on the training set only."""
    gs = GridSearchCV(
        models.tfidf_pipeline(),
        {"clf__C": [0.5, 1, 2, 4, 8, 16]},
        scoring="f1_macro",
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        n_jobs=-1,
    )
    gs.fit(list(texts), list(labels))
    return float(gs.best_params_["clf__C"]), float(gs.best_score_)


def run(out: Path, transformers: bool = True, data_dir: Path = data.DATA_DIR, limit=None) -> dict:
    train, test, dropped = data.load_splits(data_dir)
    if limit:
        test = test.sample(n=min(limit, len(test)), random_state=SEED).reset_index(drop=True)
    c, cv = tune_tfidf(train.text, train.label)
    tfidf = models.TfidfLogReg(c).fit(train.text, train.label)
    zoo = [
        models.Majority().fit(train.text, train.label),
        models.Vader(),
        models.FinVader(),
        tfidf,
    ]
    if transformers:
        zoo += [models.finbert(), models.twitter_roberta()]

    texts, y = list(test.text), list(test.label)
    results, preds = [], {}
    for m in zoo:
        m.predict(texts[:4])  # warm up (model download / lazy load) outside the timer
        t0 = time.perf_counter()
        p = m.predict(texts)
        ms = 1000 * (time.perf_counter() - t0) / len(texts)
        r = {"model": m.name, **score(y, p), "macro_f1_ci95": bootstrap_ci(y, p), "ms_per_text": ms}
        results.append(r)
        preds[m.name] = p
        print(f"{m.name:32s} macro-F1 {r['macro_f1']:.3f}  acc {r['accuracy']:.3f}  {ms:.2f} ms")

    meta = {
        "dataset": f"{data.REPO}@{data.REVISION[:7]}",
        "train_rows": len(train),
        "test_rows": len(test),
        "test_dropped_as_near_duplicates_of_train": dropped,
        "test_class_counts": test.label.value_counts().to_dict(),
        "tfidf_C": c,
        "tfidf_cv_macro_f1": cv,
        "seed": SEED,
        "hardware": f"{platform.machine()} CPU, {platform.python_version()}",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    test.assign(**{f"pred::{k}": v for k, v in preds.items()}).to_csv(
        out / "predictions.csv", index=False
    )
    return {"meta": meta, "results": results}
