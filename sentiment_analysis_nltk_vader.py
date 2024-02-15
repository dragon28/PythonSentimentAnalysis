import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import click

# Initialize the VADER sentiment intensity analyzer
sia = SentimentIntensityAnalyzer()

def get_key_with_max_value(data_dict):
    # Find the key with the maximum value
    max_key = max(data_dict, key=data_dict.get)
    return max_key

def analyze_sentiment(news):
    # Analyze the sentiment of the news
    sentiment = sia.polarity_scores(news)

    sentiments_classification = {"positive": sentiment["pos"], "negative": sentiment["neg"], "neutral": sentiment["neu"] }
    
    # Print the sentiment scores
    max_key = get_key_with_max_value(sentiments_classification)
    
    print("Sentiment: " + str(max_key) + " with score: " + str(sentiments_classification[max_key]))
    print(sentiment)


@click.command()
@click.argument('input_string')
def sentiment_analysis_result(input_string):
    
    analyze_sentiment(input_string)
    

if __name__ == '__main__':
    
    sentiment_analysis_result()
