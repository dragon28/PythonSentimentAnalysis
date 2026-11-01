"""Twitter Financial News Sentiment (zeroshot, MIT licence), pinned to one revision."""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import pandas as pd

from finsent import LABELS

REPO = "zeroshot/twitter-financial-news-sentiment"
REVISION = "ccbe24de388e287beb92dd393a335c376b350ac3"
FILES = {"train": "sent_train.csv", "test": "sent_valid.csv"}
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_URL = re.compile(r"https?://\S+")


def download(data_dir: Path = DATA_DIR) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES.values():
        path = data_dir / name
        if not path.exists():
            url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}"
            urllib.request.urlretrieve(url, path)


def normalise(text: str) -> str:
    """Key used to find near-duplicates: lowercase, URLs removed, whitespace collapsed."""
    return " ".join(_URL.sub("", text).lower().split())


def load(split: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = pd.read_csv(data_dir / FILES[split])
    df["label"] = df["label"].map(dict(enumerate(LABELS)))
    return df


def load_splits(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Train and test, with test texts that also appear in train (after normalising) removed."""
    train, test = load("train", data_dir), load("test", data_dir)
    seen = set(train.text.map(normalise))
    dup = test.text.map(normalise).isin(seen)
    return train, test[~dup].reset_index(drop=True), int(dup.sum())
