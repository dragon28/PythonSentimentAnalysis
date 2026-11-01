from pathlib import Path

from finsent import LABELS, data, models
from finsent.evaluate import bootstrap_ci, run, score

FIX = Path(__file__).parent / "fixtures"


def test_labels_map_to_names():
    df = data.load("test", FIX)
    assert set(df.label) == set(LABELS)


def test_normalise_strips_urls_and_case():
    assert data.normalise("Hi  THERE https://t.co/x ") == "hi there"


def test_near_duplicates_removed_from_test(tmp_path):
    for f in data.FILES.values():
        (tmp_path / f).write_text((FIX / f).read_text())
    with open(tmp_path / "sent_valid.csv", "a") as fh:
        fh.write('"SHARES SURGE AFTER RECORD PROFIT t0 https://t.co/a",1\n')
    _, test, dropped = data.load_splits(tmp_path)
    assert dropped == 1 and len(test) == 12


def test_compound_thresholds():
    assert models.from_compound(0.05) == "bullish"
    assert models.from_compound(-0.05) == "bearish"
    assert models.from_compound(0.0) == "neutral"


def test_score_perfect_and_ci():
    y = ["bullish", "bearish", "neutral"] * 10
    s = score(y, y)
    assert s["macro_f1"] == 1.0 and s["accuracy"] == 1.0
    assert bootstrap_ci(y, y, n=50) == (1.0, 1.0)


def test_end_to_end_offline(tmp_path):
    out = run(tmp_path, transformers=False, data_dir=FIX)
    names = [r["model"] for r in out["results"]]
    assert names[0] == "Majority class" and "TF-IDF + logistic regression" in names
    assert (tmp_path / "metrics.json").exists() and (tmp_path / "predictions.csv").exists()
    assert out["meta"]["test_rows"] == 12
