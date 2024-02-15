import nltk
from finvader import finvader
import click

def analyze_sentiment(news):
    
    compound_score = finvader(news, use_sentibignomics = True, use_henry = True, indicator = 'compound')
    
    print("Sentiment score: " + str(compound_score))
    

@click.command()
@click.argument('input_string')
def sentiment_analysis_result(input_string):
    
    analyze_sentiment(input_string)
    

if __name__ == '__main__':
    
    sentiment_analysis_result()