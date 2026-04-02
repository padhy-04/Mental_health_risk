
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import scipy.sparse as sp
import os

app = Flask(__name__)
CORS(app)

# Load models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH   = os.path.join(BASE_DIR, "models", "models.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "features.pkl")

print("Loading models...")
with open(MODELS_PATH, "rb") as f:
    model_data = pickle.load(f)
with open(FEATURES_PATH, "rb") as f:
    feat_data = pickle.load(f)

rf_model = model_data["rf_model"]
le       = model_data["le"]
tfidf    = feat_data["tfidf"]

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
print("Models loaded successfully!")

def clean_text(text):
    text  = str(text).lower()
    text  = re.sub(r"http\S+|www\S+", "", text)
    text  = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words
             if w not in stop_words and len(w) > 2]
    return " ".join(words)

def get_volatility(text):
    words  = text.split()[:20]
    if len(words) < 2:
        return 0.0
    scores = [TextBlob(w).sentiment.polarity for w in words]
    return float(np.std(scores))

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data      = request.get_json()
        user_text = data.get("text", "")
        is_night  = data.get("is_night", False)
        freq      = float(data.get("posts_per_day", 5))

        if not user_text.strip():
            return jsonify({"error": "No text provided"}), 400

        # Clean text
        cleaned = clean_text(user_text)

        # TF-IDF features
        tfidf_feat = tfidf.transform([cleaned])

        # Sentiment features
        blob       = TextBlob(cleaned)
        polarity   = float(blob.sentiment.polarity)
        subjectivity = float(blob.sentiment.subjectivity)

        # Behavioral features
        volatility = get_volatility(cleaned)
        post_len   = len(cleaned.split())
        night      = 0.82 if is_night else 0.18

        # Combine features
        behavioral = sp.csr_matrix([[
            polarity, subjectivity,
            volatility, post_len,
            freq, night, -polarity
        ]])
        X_input = sp.hstack([tfidf_feat, behavioral])

        # Real ML prediction
        prediction  = rf_model.predict(X_input)[0]
        probability = rf_model.predict_proba(X_input)[0]
        risk_label  = le.classes_[prediction]
        confidence  = float(max(probability) * 100)

        # Composite score
        composite = min(max(
            (1 - polarity) * 30 +
            min(volatility * 100, 25) +
            night * 20 +
            min(post_len / 100 * 15, 15) +
            min(freq / 20 * 10, 10),
            0), 100)

        return jsonify({
            "risk_level"       : risk_label,
            "confidence"       : round(confidence, 1),
            "composite_score"  : round(composite, 1),
            "sentiment_polarity": round(polarity, 4),
            "emotional_volatility": round(volatility, 4),
            "subjectivity"     : round(subjectivity, 4),
            "post_length"      : post_len,
            "night_activity"   : round(night, 2),
            "posting_frequency": freq,
            "cleaned_text"     : cleaned
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status"  : "running",
        "model"   : "Random Forest",
        "accuracy": "93.36%",
        "features": 507
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
