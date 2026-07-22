import json
import math
import os
import pickle
import threading
import time

from flask import Flask, jsonify, render_template, request

from src.fetch_article import ArticleFetchError, fetch_article_text
from src.live_feed import fetch_live_headlines
from src.preprocess import clean_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# How often the Live Wire re-pulls and re-classifies RSS headlines.
LIVE_REFRESH_SECONDS = 120

app = Flask(__name__)

vectorizer = None
model = None
metrics = {}

_live_cache = {"items": [], "updated": 0}
_live_lock = threading.Lock()


def load_artifacts():
    global vectorizer, model, metrics
    vec_path = os.path.join(MODELS_DIR, "vectorizer.pkl")
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")

    if os.path.exists(vec_path) and os.path.exists(model_path):
        with open(vec_path, "rb") as f:
            vectorizer = pickle.load(f)
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)


load_artifacts()


def classify_text(raw_text: str):
    """Run the shared preprocessing + vectorizer + model pipeline on a
    piece of text. Returns (label, confidence) where label is
    'real' | 'fake' and confidence is a 0-100 float or None."""
    cleaned = clean_text(raw_text)
    vec = vectorizer.transform([cleaned])

    label = int(model.predict(vec)[0])  # 0 = real, 1 = fake

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        confidence = round(float(max(proba)) * 100, 2)
    elif hasattr(model, "decision_function"):
        score = model.decision_function(vec)[0]
        confidence = round(100 / (1 + math.exp(-abs(score))), 2)

    return ("fake" if label == 1 else "real"), confidence


def refresh_live_cache():
    """Pull fresh headlines from the RSS feeds and classify each one.
    Safe to call even if the model hasn't been trained yet (items just
    come back unlabeled)."""
    headlines = fetch_live_headlines()
    results = []
    for h in headlines:
        combined = f"{h['title']}. {h.get('summary', '')}".strip()
        if model is not None and vectorizer is not None and combined:
            label, confidence = classify_text(combined)
        else:
            label, confidence = "unknown", None
        results.append({**h, "label": label, "confidence": confidence})

    with _live_lock:
        _live_cache["items"] = results
        _live_cache["updated"] = time.time()


@app.route("/")
def home():
    best_model_name = metrics.get("best_model", "N/A")
    best_f1 = None
    if metrics.get("results") and best_model_name in metrics["results"]:
        best_f1 = metrics["results"][best_model_name]["f1"]
    return render_template("index.html", best_model=best_model_name, best_f1=best_f1)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not found. Run `python src/train_models.py` first."}), 500

    data = request.get_json(force=True) or {}
    url = (data.get("url") or "").strip()
    text = (data.get("text") or "").strip()
    source_title = None

    if url:
        try:
            article = fetch_article_text(url)
        except ArticleFetchError as e:
            return jsonify({"error": f"Could not fetch that URL: {e}"}), 400

        text = article["text"]
        source_title = article["title"]
        if not text or len(text.split()) < 20:
            return jsonify({
                "error": "Couldn't pull enough article text from that page "
                         "(it may be paywalled or JS-rendered). Try pasting "
                         "the text directly instead."
            }), 400
    elif not text:
        return jsonify({"error": "Please enter some text or a URL to analyze."}), 400

    label, confidence = classify_text(text)

    return jsonify({
        "label": label,
        "confidence": confidence,
        "title": source_title,
        "word_count": len(text.split()),
    })


@app.route("/live_feed")
def live_feed():
    """Real-time news wire: returns the latest classified headlines,
    refreshing the underlying RSS pull in the background whenever the
    cache goes stale (so requests stay fast and feeds aren't hammered
    on every poll)."""
    with _live_lock:
        stale = (time.time() - _live_cache["updated"]) > LIVE_REFRESH_SECONDS

    if stale:
        refresh_live_cache()

    with _live_lock:
        return jsonify({
            "items": _live_cache["items"],
            "updated": _live_cache["updated"],
            "refresh_seconds": LIVE_REFRESH_SECONDS,
        })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
