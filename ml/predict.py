import pickle
import string
import nltk
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ml", "model.pkl")

# Load NLP resources
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    """
    Applies the same preprocessing steps used during training
    to a single input text.
    """

    # 1. Lowercase
    text = text.lower()

    # 2. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 3. Tokenize
    tokens = word_tokenize(text)

    # 4. Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # 5. Join tokens
    clean_text = " ".join(tokens)

    return clean_text

def predict_spam(text):
    """
    Loads the trained model and vectorizer
    and predicts whether the input text is spam or not.
    """

    # Load model and vectorizer
    with open(MODEL_PATH, "rb") as f:
        model, vectorizer = pickle.load(f)

    # Preprocess input text
    clean_text = preprocess_text(text)

    # Vectorize input text
    vectorized_text = vectorizer.transform([clean_text])

    # Predict
    prediction = model.predict(vectorized_text)[0]

    return "Spam" if prediction == 1 else "Not Spam"