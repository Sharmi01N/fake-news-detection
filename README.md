# The Verdict — Real-Time Fake News Detection (ECE 3100)

A machine-learning web app that classifies news as **real** or **fake** in
real time, built for the ECE 3100 (Software Development Project – II)
proposal *"Fake News Detection Using Machine Learning & Deep Learning."*

Three ways to get a verdict:

1. **Paste text** — a headline, excerpt, or claim, classified instantly.
2. **Check a live URL** — paste a link to any news article; the server
   fetches the page *right then*, extracts the story text, and classifies
   it — no manual copy-pasting needed.
3. **Live Wire** — a ticker on the homepage that continuously pulls
   headlines from public news RSS feeds and classifies each one
   automatically, refreshing in the background so the page always shows
   current, freshly-scored news.

## What's inside

```
fake-news-detection/
├── app.py                     # Flask app: front page + /predict + /live_feed
├── requirements.txt
├── data/
│   ├── generate_sample_data.py   # builds a small synthetic demo dataset
│   └── sample_data.csv           # synthetic demo data (swap for the real dataset)
├── src/
│   ├── preprocess.py          # shared text-cleaning pipeline
│   ├── train_models.py        # trains 5 ML models, saves the best one
│   ├── fetch_article.py       # fetches + extracts article text from a URL
│   └── live_feed.py           # pulls headlines from public RSS feeds
├── models/                    # created after training (vectorizer + model + metrics.json)
├── templates/index.html       # front page
└── static/style.css, script.js
```

## Quick start (VS Code / local machine)

```bash
# 1. clone your repo and enter it
git clone <your-repo-url>
cd fake-news-detection

# 2. create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 3. install dependencies
pip install -r requirements.txt

# 4. train the models (uses data/sample_data.csv by default)
python src/train_models.py

# 5. run the app
python app.py
```

Open **http://127.0.0.1:5000**. By default you land on the **Paste text** tab —
type or paste a headline/excerpt and click **"Send to the desk."** Switch to
**Check a live URL** to paste a news article link instead; the server fetches
and reads it live. Scroll down to **Live Wire** to watch real RSS headlines
get classified automatically as they arrive.

## Dataset

`data/sample_data.csv` is built from three real sources merged by
`data/build_dataset.py`:

- **ISOT** `True.csv` / `Fake.csv` — 21,417 real / 23,481 fake news articles
- **GlobalFakeNews_Research2026_v1.csv** — 1,200 labeled samples

After cleaning and de-duplication: **39,117 samples** (21,203 real / 17,914 fake).

To rebuild from raw sources: place `True.csv`, `Fake.csv`, and
`GlobalFakeNews_Research2026_v1.csv` in `data/`, then run:

```bash
python data/build_dataset.py
python src/train_models.py
```

## How it works

1. **Preprocessing** (`src/preprocess.py`): lowercases text, strips HTML/URLs/punctuation,
   removes stopwords, lemmatizes — the exact same function is reused at both
   training and prediction time so there's no train/serve skew.
2. **Feature engineering**: TF-IDF vectorizer (unigrams + bigrams, 5000 features).
3. **Models trained** (`src/train_models.py`): Logistic Regression, Naive Bayes,
   Linear SVM, Random Forest, and XGBoost. Each is scored on accuracy,
   precision, recall, and F1; the best F1 wins and is saved to `models/best_model.pkl`.
4. **Serving** (`app.py`): loads the saved vectorizer + model and exposes
   `POST /predict`, which accepts **either**:
   - `{"text": "..."}` → classified directly, or
   - `{"url": "https://..."}` → the server fetches the page live
     (`src/fetch_article.py`, using `requests` + `BeautifulSoup`, extracting
     the `<article>` body or the largest cluster of `<p>` tags plus an
     `og:title`/`<title>` fallback), then classifies the extracted text.

   Response either way: `{"label": "real"|"fake", "confidence": 0-100,
   "title": "..."|null, "word_count": N}`.

5. **Real-time monitoring** (`GET /live_feed`, `src/live_feed.py`): pulls
   current headlines from a small set of public RSS feeds (BBC, Al Jazeera,
   NPR, The Daily Star), classifies each one with the same pipeline, and
   caches the results server-side for `LIVE_REFRESH_SECONDS` (120s by
   default) so a feed outage or slow RSS server doesn't block requests. The
   front-end polls this endpoint every 30 seconds and re-renders the **Live
   Wire** ticker with a REAL/FAKE badge and confidence per headline.

   Note: these feeds are legitimate outlets used to demonstrate continuous,
   automatic classification — expect the ticker to mostly show "REAL," since
   the point is showing the pipeline running live on live traffic, not that
   these sources publish fake news. Add more feeds (e.g. Bangladeshi
   sources) or plug in a fact-check/claims API in `FEEDS` inside
   `src/live_feed.py` if you want more variety for the demo.

## Deploying (GitHub + free hosting)

1. Push this folder to a GitHub repo.
2. Add a `Procfile` with `web: gunicorn app:app` and add `gunicorn` to
   `requirements.txt` if deploying to Render / Railway / Heroku-style hosts.
3. Make sure `models/` (the trained `.pkl` files) is committed, or run the
   training step as part of your deploy/build script.
4. Set the host's start command to `python app.py` (or `gunicorn app:app`
   for production) and the port to the platform's assigned `$PORT`.

## Extending toward the deep-learning models in the proposal

The proposal also lists CNN, LSTM/GRU, and BERT. To add those:
- Build a `src/train_deep_models.py` using TensorFlow/Keras or PyTorch,
  tokenizing with Keras `Tokenizer` or a Hugging Face tokenizer.
- Save the trained model + tokenizer the same way `train_models.py` saves
  the TF-IDF model, then add a toggle in `app.py` to choose which model
  serves predictions.

## Results

Trained on the merged 39,117-sample dataset:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.9849 | 0.9890 | 0.9780 | 0.9834 |
| Naive Bayes | 0.9477 | 0.9411 | 0.9450 | 0.9430 |
| SVM (Linear) | 0.9932 | 0.9933 | 0.9919 | 0.9926 |
| **Random Forest (best)** | **0.9967** | **0.9978** | **0.9950** | **0.9964** |
| XGBoost | 0.9965 | 0.9980 | 0.9944 | 0.9962 |

Note: the ISOT portion of the dataset is a well-known benchmark where real
articles mostly carry a Reuters wire-service dateline (e.g. `WASHINGTON
(Reuters) -`) and fake ones don't. The model partly keys off that
formatting pattern, not just content, so it's confident on ISOT-style text
but less reliable on short, generic sentences without that structure. This
is a documented limitation of the ISOT dataset itself, not a bug in the
pipeline.

## Notes for the report / demo

- `models/metrics.json` (created after training) has the accuracy /
  precision / recall / F1 for every model trained — use this table directly
  in your report's "Results" section.
- The front page shows which model is currently "on duty" and its
  validation F1 score, pulled live from `metrics.json`.
