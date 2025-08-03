# predict_vader.py

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# ensure we have the lexicon
nltk.download('vader_lexicon')

def vader_predict(text: str) -> str:
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(text)
    # compound >= 0.05 → positive; <= -0.05 → negative; else neutral
    return 'Positive' if scores['compound'] >= 0.05 else 'Negative'

if __name__ == '__main__':
    text = input("Enter your review: ")
    print(vader_predict(text))
