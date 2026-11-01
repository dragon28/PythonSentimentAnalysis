"""Every model maps a list of texts to labels in LABELS. Transformers load lazily."""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from finsent import LABELS

# Standard VADER cut-offs from the original paper; not tuned on the test set.
POS, NEG = 0.05, -0.05


def from_compound(score: float) -> str:
    return "bullish" if score >= POS else "bearish" if score <= NEG else "neutral"


class Majority:
    name = "Majority class"

    def fit(self, texts, labels):
        self.label = Counter(labels).most_common(1)[0][0]
        return self

    def predict(self, texts):
        return [self.label] * len(texts)


class Vader:
    name = "VADER (NLTK lexicon)"

    def __init__(self):
        import nltk

        nltk.download("vader_lexicon", quiet=True)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        self.sia = SentimentIntensityAnalyzer()

    def predict(self, texts):
        return [from_compound(self.sia.polarity_scores(t)["compound"]) for t in texts]


class FinVader:
    name = "FinVADER (finance lexicon)"

    def predict(self, texts):
        from finvader import finvader

        return [
            from_compound(
                finvader(t, use_sentibignomics=True, use_henry=True, indicator="compound")
            )
            for t in texts
        ]


def tfidf_pipeline(c: float = 4.0) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", LogisticRegression(C=c, max_iter=2000, class_weight="balanced")),
        ]
    )


class TfidfLogReg:
    name = "TF-IDF + logistic regression"

    def __init__(self, c: float = 4.0):
        self.pipe = tfidf_pipeline(c)

    def fit(self, texts, labels):
        self.pipe.fit(list(texts), list(labels))
        return self

    def predict(self, texts):
        return list(self.pipe.predict(list(texts)))

    def predict_proba(self, texts):
        p = self.pipe.predict_proba(list(texts))
        return [dict(zip(self.pipe.classes_, row, strict=True)) for row in p]


class HFModel:
    """Off-the-shelf Hugging Face classifier, pinned to a revision; labels mapped to ours."""

    MAP = {"positive": "bullish", "negative": "bearish", "neutral": "neutral"}

    def __init__(self, repo: str, revision: str, name: str, batch_size: int = 32):
        self.repo, self.revision, self.name, self.batch_size = repo, revision, name, batch_size
        self._tok = self._model = None

    def _load(self):
        if self._model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.repo, revision=self.revision)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.repo, revision=self.revision
            ).eval()
            id2 = self._model.config.id2label
            self._labels = [self.MAP[id2[i].lower()] for i in range(len(id2))]

    def predict_proba(self, texts):
        import torch

        self._load()
        out = []
        for i in range(0, len(texts), self.batch_size):
            enc = self._tok(
                list(texts[i : i + self.batch_size]),
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            with torch.no_grad():
                p = torch.softmax(self._model(**enc).logits, dim=-1).numpy()
            out += [dict(zip(self._labels, row.tolist(), strict=True)) for row in p]
        return out

    def predict(self, texts):
        return [max(p, key=p.get) for p in self.predict_proba(texts)]


def finbert() -> HFModel:
    return HFModel(
        "ProsusAI/finbert", "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031", "FinBERT (ProsusAI)"
    )


def twitter_roberta() -> HFModel:
    return HFModel(
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7",
        "Twitter-RoBERTa (CardiffNLP)",
    )


def check_labels() -> None:
    assert set(LABELS) == set(HFModel.MAP.values())
    assert np.isclose(POS, -NEG)
