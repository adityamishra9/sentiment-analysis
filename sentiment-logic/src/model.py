# src/model.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
import joblib

def train_and_save(df, model_path='models/sentiment_model.pkl'):
    vec = TfidfVectorizer(max_features=5_000, ngram_range=(1,3))
    X = vec.fit_transform(df['review'])
    y = df['sentiment']
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)
    joblib.dump((vec, clf), model_path)
    print(f"Model saved to {model_path}")
