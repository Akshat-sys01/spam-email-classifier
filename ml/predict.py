import pickle
import string
import os
os.environ["NLTK_DATA"] = "/opt/render/nltk_data"

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ml", "model.pkl")

# --------------------------------------------------
# Load model & vectorizer ONCE (IMPORTANT)
# --------------------------------------------------
with open(MODEL_PATH, "rb") as f:
    model, vectorizer = pickle.load(f)

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------
stop_words = set(stopwords.words("english"))

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
    clean_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([clean_text])
    prediction = model.predict(vectorized_text)[0]
    return "Spam" if prediction == 1 else "Not Spam"
