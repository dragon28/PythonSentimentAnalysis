"""Streamlit demo: try a headline, and browse the saved benchmark.  `uv run streamlit run app.py`"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from finsent import data, models

ROOT = Path(__file__).parent
st.set_page_config(page_title="Financial sentiment lab", layout="wide")
st.title("Financial-news sentiment: lexicons vs classic ML vs transformers")


@st.cache_resource(show_spinner="Loading data and training TF-IDF model (first start only)...")
def light_models():
    data.download()
    m = json.loads((ROOT / "results/metrics.json").read_text())
    train, _, _ = data.load_splits()
    tfidf = models.TfidfLogReg(m["meta"]["tfidf_C"]).fit(train.text, train.label)
    return [models.Vader(), models.FinVader(), tfidf]


try_it, bench = st.tabs(["Try a headline", "Benchmark results"])

with try_it:
    text = st.text_area(
        "Financial headline or tweet (max 500 characters; not stored)",
        "Oil prices extend losses on US inventory shock, demand concerns",
        max_chars=500,
    )
    if text.strip():
        rows = [{"model": m.name, "prediction": m.predict([text])[0]} for m in light_models()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "FinBERT and Twitter-RoBERTa are not loaded here to stay within free hosting memory; "
        "their scores are in the Benchmark tab, and `finsent predict` runs them locally."
    )

with bench:
    m = json.loads((ROOT / "results/metrics.json").read_text())
    meta = m["meta"]
    st.write(
        f"Test set: {meta['test_rows']:,} labelled financial tweets ({meta['dataset']}), "
        f"never used for training or tuning."
    )
    df = pd.DataFrame(
        [
            {
                "model": r["model"],
                "macro-F1": round(r["macro_f1"], 3),
                "95% CI": f"{r['macro_f1_ci95'][0]:.3f}-{r['macro_f1_ci95'][1]:.3f}",
                "accuracy": round(r["accuracy"], 3),
                "ms/text": round(r["ms_per_text"], 2),
            }
            for r in m["results"]
        ]
    ).sort_values("macro-F1", ascending=False)
    st.dataframe(df, hide_index=True, width="stretch")
    st.bar_chart(df.set_index("model")["macro-F1"])
    with st.expander("Full report: confusion matrices and error examples"):
        st.markdown((ROOT / "results/report.md").read_text())
