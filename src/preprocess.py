"""
preprocess.py
-------------
Text cleaning / preprocessing pipeline used by both training and inference,
so the exact same steps are applied at train time and predict time.
"""
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for pkg in ["stopwords", "wordnet", "omw-1.4", "punkt"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

_lemmatizer = WordNetLemmatizer()
_stopwords = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Lowercase, strip HTML/URLs/punctuation/numbers, tokenize,
    remove stopwords, and lemmatize. Returns a cleaned string."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)                 # HTML tags
    text = re.sub(r"http\S+|www\.\S+", " ", text)       # URLs
    text = re.sub(r"[^a-z\s]", " ", text)               # keep letters only
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [t for t in tokens if t not in _stopwords and len(t) > 2]
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


if __name__ == "__main__":
    sample = "BREAKING!! Doctors HATE this <b>one trick</b> — visit http://scam.example.com now!!"
    print("Original :", sample)
    print("Cleaned  :", clean_text(sample))
