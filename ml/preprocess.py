import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLP resources (only once)
# nltk.download('punkt')
# nltk.download('stopwords')

def load_and_clean_data(csv_path):
    """
    Loads the spam dataset and performs text preprocessing.
    Returns:
    X -> TF-IDF feature matrix
    y -> Labels (0 = ham, 1 = spam)
    vectorizer -> trained TF-IDF vectorizer
    """

    # 1. Load dataset
    df = pd.read_csv(
        csv_path,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="latin-1"
    )

    # 3. Encode labels
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # 4. Convert text to lowercase
    df['message'] = df['message'].str.lower()

    # 5. Remove punctuation
    def remove_punctuation(text):
        return text.translate(str.maketrans('', '', string.punctuation))
    
    df['message'] = df['message'].apply(remove_punctuation)

    # 6. Tokenization
    df['tokens'] = df['message'].apply(word_tokenize)

    # 7. Remove stopwords
    stop_words = set(stopwords.words('english'))

    df['tokens'] = df['tokens'].apply(
        lambda words: [word for word in words if word not in stop_words]
    )

    # 8. Join tokens back into text
    df['clean_text'] = df['tokens'].apply(lambda x: " ".join(x))

    # 9. TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=3000)

    X = vectorizer.fit_transform(df['clean_text'])
    y = df['label']

    return X, y, vectorizer