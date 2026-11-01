"""Turn results/metrics.json + predictions.csv into results/report.md."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from finsent import LABELS


def build(results: Path = Path("results"), n_examples: int = 12) -> str:
    m = json.loads((results / "metrics.json").read_text())
    meta, rows = m["meta"], m["results"]
    lines = [
        "# Results",
        "",
        f"Dataset `{meta['dataset']}`: {meta['train_rows']:,} train tweets, "
        f"{meta['test_rows']:,} test tweets ({meta['test_dropped_as_near_duplicates_of_train']} "
        "test tweets removed as near-duplicates of training tweets). "
        f"Test classes: {meta['test_class_counts']}.",
        "",
        "| Model | Macro-F1 (95% CI) | Accuracy | F1 bearish | F1 bullish | F1 neutral | ms/text |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: -r["macro_f1"]):
        lo, hi = r["macro_f1_ci95"]
        pc = r["per_class"]
        lines.append(
            f"| {r['model']} | {r['macro_f1']:.3f} ({lo:.3f}-{hi:.3f}) | {r['accuracy']:.3f} | "
            + " | ".join(f"{pc[c]['f1']:.2f}" for c in LABELS)
            + f" | {r['ms_per_text']:.2f} |"
        )
    lines += [
        "",
        f"TF-IDF C={meta['tfidf_C']:g} chosen by 5-fold CV on train only "
        f"(CV macro-F1 {meta['tfidf_cv_macro_f1']:.3f}). Timing: single process on "
        f"{meta['hardware']}, batch of all test tweets, after warm-up.",
        "",
        "## Confusion matrices (rows = true, columns = predicted: " + ", ".join(LABELS) + ")",
        "",
    ]
    for r in rows:
        lines.append(f"- {r['model']}: `{r['confusion']}`")

    pred_file = results / "predictions.csv"
    if pred_file.exists():
        p = pd.read_csv(pred_file)
        best = max(rows, key=lambda r: r["macro_f1"])["model"]
        wrong = p[p[f"pred::{best}"] != p.label].sample(n=min(n_examples, len(p)), random_state=0)
        lines += [
            "",
            f"## Random sample of {best} errors",
            "",
            "| Tweet | True | Predicted | FinBERT |",
            "|---|---|---|---|",
        ]
        fb = "pred::FinBERT (ProsusAI)"
        for _, w in wrong.iterrows():
            t = str(w.text).replace("|", "/").replace("\n", " ")[:140]
            lines.append(f"| {t} | {w.label} | {w[f'pred::{best}']} | {w.get(fb, '')} |")
    text = "\n".join(lines) + "\n"
    (results / "report.md").write_text(text)
    return text
