from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple

_vectorizer = None

def score_similarity(text_a: str, text_b: str) -> float:
    """Return similarity score in range [0,1] between two texts using TF-IDF + cosine."""
    global _vectorizer
    if _vectorizer is None:
        _vectorizer = TfidfVectorizer(stop_words='english')
    docs = [text_a or "", text_b or ""]
    tfidf = _vectorizer.fit_transform(docs)
    sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0,0]
    return float(sim)
