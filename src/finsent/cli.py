"""finsent download | evaluate | predict "text" """

from __future__ import annotations

from pathlib import Path

import click

from finsent import data


@click.group()
def main():
    pass


@main.command()
def download():
    """Fetch the pinned dataset into ./data (not committed)."""
    data.download()
    click.echo(f"downloaded to {data.DATA_DIR}")


@main.command()
@click.option("--out", default="results", type=click.Path(path_type=Path))
@click.option("--no-transformers", is_flag=True, help="skip FinBERT / RoBERTa (fast, offline)")
@click.option("--limit", type=int, help="evaluate on a random subset (smoke test)")
def evaluate(out, no_transformers, limit):
    """Score all models on the same held-out set; write metrics + predictions."""
    from finsent.evaluate import run

    data.download()
    run(out, transformers=not no_transformers, limit=limit)
    from finsent.report import build

    build(out)
    click.echo(f"wrote {out}/metrics.json and {out}/report.md")


@main.command()
@click.argument("text")
def predict(text):
    """Show every model's label for one headline."""
    from finsent import models

    train, _, _ = data.load_splits()
    zoo = [
        models.Vader(),
        models.FinVader(),
        models.TfidfLogReg().fit(train.text, train.label),
        models.finbert(),
        models.twitter_roberta(),
    ]
    for m in zoo:
        click.echo(f"{m.name:32s} {m.predict([text])[0]}")


if __name__ == "__main__":
    main()
