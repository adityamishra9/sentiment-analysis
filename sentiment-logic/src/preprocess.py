# src/preprocess.py
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Words that trigger a negation scope
NEGATIONS = {"no", "not", "never", "n't"}
# Keep negation tokens in the text
STOP_WORDS = set(stopwords.words('english')) - NEGATIONS

def tokenize_with_negation(text, scope_size=3):
    """
    Split text into tokens, and for up to `scope_size` tokens
    after a negation word, prefix each with "NOT_".
    """
    words = text.split()
    out = []
    neg_scope = 0

    for w in words:
        clean = re.sub(r'[^\w\'’]', '', w.lower())
        if clean in NEGATIONS:
            out.append(clean)
            neg_scope = scope_size
        elif neg_scope > 0:
            out.append("NOT_" + clean)
            neg_scope -= 1
        else:
            out.append(clean)
    return out

def clean_reviews(df):
    df = df.copy()
    df['review'] = (
        df['review']
          .str.lower()
          .apply(tokenize_with_negation)
          .apply(lambda tokens: ' '.join(t for t in tokens if t not in STOP_WORDS))
    )
    df['sentiment'] = df['rating'].apply(lambda r: 1 if r >= 4 else 0)
    return df