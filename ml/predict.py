import pickle
import string
import os
import nltk

# --------------------------------------------------
# NLTK configuration (NO downloads here)
# --------------------------------------------------
NLTK_DATA_PATH = "/opt/render/nltk_data"
os.environ["NLTK_DATA"] = NLTK_DATA_PATH
nltk.data.path.append(NLTK_DATA_PATH)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ml", "model.pkl")

# --------------------------------------------------
# Lazy-loaded model
# --------------------------------------------------
model = None
vectorizer = None

def load_model():
    global model, vectorizer
    if model is None or vectorizer is None:
        with open(MODEL_PATH, "rb") as f:
            model, vectorizer = pickle.load(f)

# --------------------------------------------------
# Load stopwords ONCE (no download fallback)
# --------------------------------------------------
stop_words = set(stopwords.words("english"))

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
def predict_spam(text):
    load_model()
    clean_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([clean_text])
    prediction = model.predict(vectorized_text)[0]
    return "Spam" if prediction == 1 else "Not Spam"
