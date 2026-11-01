# Results

Dataset `zeroshot/twitter-financial-news-sentiment@ccbe24d`: 9,543 train tweets, 2,349 test tweets (39 test tweets removed as near-duplicates of training tweets). Test classes: {'neutral': 1536, 'bullish': 473, 'bearish': 340}.

| Model | Macro-F1 (95% CI) | Accuracy | F1 bearish | F1 bullish | F1 neutral | ms/text |
|---|---|---|---|---|---|---|
| TF-IDF + logistic regression | 0.767 (0.745-0.785) | 0.825 | 0.68 | 0.73 | 0.89 | 0.01 |
| FinBERT (ProsusAI) | 0.667 (0.644-0.687) | 0.724 | 0.59 | 0.61 | 0.80 | 8.52 |
| Twitter-RoBERTa (CardiffNLP) | 0.612 (0.587-0.634) | 0.702 | 0.53 | 0.52 | 0.78 | 8.63 |
| FinVADER (finance lexicon) | 0.515 (0.491-0.538) | 0.582 | 0.41 | 0.45 | 0.68 | 2.73 |
| VADER (NLTK lexicon) | 0.448 (0.426-0.469) | 0.496 | 0.40 | 0.35 | 0.59 | 0.05 |
| Majority class | 0.264 (0.259-0.268) | 0.654 | 0.00 | 0.00 | 0.79 | 0.00 |

TF-IDF C=16 chosen by 5-fold CV on train only (CV macro-F1 0.731). Timing: single process on x86_64 CPU, 3.12.13, batch of all test tweets, after warm-up.

## Confusion matrices (rows = true, columns = predicted: bearish, bullish, neutral)

- Majority class: `[[0, 0, 340], [0, 0, 473], [0, 0, 1536]]`
- VADER (NLTK lexicon): `[[165, 79, 96], [41, 217, 215], [283, 471, 782]]`
- FinVADER (finance lexicon): `[[157, 51, 132], [30, 254, 189], [233, 347, 956]]`
- TF-IDF + logistic regression: `[[235, 31, 74], [31, 345, 97], [83, 94, 1359]]`
- FinBERT (ProsusAI): `[[257, 12, 71], [57, 283, 133], [210, 166, 1160]]`
- Twitter-RoBERTa (CardiffNLP): `[[174, 2, 164], [12, 221, 240], [132, 151, 1253]]`

## Random sample of TF-IDF + logistic regression errors

| Tweet | True | Predicted | FinBERT |
|---|---|---|---|
| Stocks aim for back-to-back gains as investors focus on coronavirus case tally https://t.co/QBhdB4ipyD | bullish | bearish | bullish |
| Volkswagen talks up next-gen EV wagon | neutral | bullish | neutral |
| Global air passenger traffic has grown at 4% YoY in 2019 ytd while air freight volumes contracted 4% over the same period: BofA | bullish | neutral | bullish |
| Can $MSFT head higher?  https://t.co/V5l6NINYrr | neutral | bullish | neutral |
| Why the markets 'look clean' into 2019 year-end | bullish | neutral | neutral |
| China M&A Bankers Face Another Grim Year | bearish | neutral | bearish |
| Suncor's dividend boost reassures analysts amid weak Q4 | bullish | bearish | bullish |
| Markel Corporation Beat Analyst Estimates: See What The Consensus Is Forecasting For Next Year | bullish | neutral | neutral |
| CME : Volatility in New York gold prices impacting liquidity in London - LBMA #CME #Stock #MarketScreener… https://t.co/8ILGYoJHzS | bearish | neutral | neutral |
| $AMZN - Amazon: One Of The Best Long-Term Investment In The Tech Sector. https://t.co/zC79wifAYj #finance #investing #stocks | bullish | neutral | bullish |
| Coronavirus pandemic has driven a very 'ample' market correction: former European Central Bank chief… https://t.co/dSD0FhQt36 | bearish | bullish | bearish |
| Dow- Chemically Speaking | neutral | bullish | neutral |
