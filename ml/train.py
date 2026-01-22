from preprocess import load_and_clean_data
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# 1. Load and preprocess data
X, y, vectorizer = load_and_clean_data(r"C:\Academics 25-26\4th Sem\My Work\Projects\spam_email_classifier\ml\data\SMSSpamCollection")

# 2. Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Initialize Naive Bayes model
model = MultinomialNB()

# 4. Train the model
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Save the model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)

print("\nModel saved as model.pkl")