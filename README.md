# Python Sentiment Analysis

PLease make sure that you have installed Python version 3.10

Requirements:

NLTK:

`pip install nltk`

or

`pip3 install nltk`

PyTorch [CPU only]:

Please refer to [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) for more information on how to install PyTorch [CPU Only].

Transformers [CPU Only]:

`pip install 'transformers[torch]'`

or

`pip3 install 'transformers[torch]'`

Example of Usage:

Refering to [https://www.investing.com/news/commodities-news/oil-prices-extend-losses-on-us-inventory-shock-demand-concerns-3304210](https://www.investing.com/news/commodities-news/oil-prices-extend-losses-on-us-inventory-shock-demand-concerns-3304210) :

<h1>Oil prices extend losses on US inventory shock, demand concerns</h1>

<p>Investing.com-- Oil prices fell in Asian trade on Thursday, extending losses after a substantially bigger-than-expected build in U.S. inventories pointed to well-supplied markets, while Japanese recession signals drove up concerns over slowing demand. Crude prices had lost over $1 each on Wednesday after data showed U.S. oil inventories grew a staggering 12 million barrels in the week to February 9, much higher than expectations for a build of 3.3 million barrels. The reading was driven chiefly by record-high U.S. production, indicating that the world’s largest fuel consumer remained well-supplied with oil. While gasoline and distillate inventories shrank, the drop was attributed largely to extended refinery shutdowns, due to maintenance activity. U.S. fuel demand was seen weakening in recent months amid adverse weather and increasing economic pressure from high inflation and interest rates.</p>

```
$ python sentiment_analysis_nltk_vader.py "Oil prices extend losses on US inventory shock, demand concerns"
Sentiment: negative with score: 0.469
{'neg': 0.469, 'neu': 0.414, 'pos': 0.117, 'compound': -0.6249}

$ python sentiment_analysis_nltk_vader.py "Investing.com-- Oil prices fell in Asian trade on Thursday, extending losses after a substantially bigger-than-expected build in U.S. inventories pointed to well-supplied markets, while Japanese recession signals drove up concerns over slowing demand. Crude prices had lost over $1 each on Wednesday after data showed U.S. oil inventories grew a staggering 12 million barrels in the week to February 9, much higher than expectations for a build of 3.3 million barrels. The reading was driven chiefly by record-high U.S. production, indicating that the world’s largest fuel consumer remained well-supplied with oil. While gasoline and distillate inventories shrank, the drop was attributed largely to extended refinery shutdowns, due to maintenance activity. U.S. fuel demand was seen weakening in recent months amid adverse weather and increasing economic pressure from high inflation and interest rates."
Sentiment: neutral with score: 0.815
{'neg': 0.164, 'neu': 0.815, 'pos': 0.021, 'compound': -0.9485}
```

```
$ python sentiment_analysis_transformers_finbert.py "Oil prices extend losses on US inventory shock, demand concerns"
Sentiment: negative with score: 0.9329016804695129
Positive: 0.029342176392674446
Negative: 0.9329016804695129
Neutral: 0.03775615245103836

$ python sentiment_analysis_transformers_finbert.py "Investing.com-- Oil prices fell in Asian trade on Thursday, extending losses after a substantially bigger-than-expected build in U.S. inventories pointed to well-supplied markets, while Japanese recession signals drove up concerns over slowing demand. Crude prices had lost over $1 each on Wednesday after data showed U.S. oil inventories grew a staggering 12 million barrels in the week to February 9, much higher than expectations for a build of 3.3 million barrels. The reading was driven chiefly by record-high U.S. production, indicating that the world’s largest fuel consumer remained well-supplied with oil. While gasoline and distillate inventories shrank, the drop was attributed largely to extended refinery shutdowns, due to maintenance activity. U.S. fuel demand was seen weakening in recent months amid adverse weather and increasing economic pressure from high inflation and interest rates."
Sentiment: negative with score: 0.9738574028015137
Positive: 0.011814839206635952
Negative: 0.9738574028015137
Neutral: 0.014327661134302616
```
