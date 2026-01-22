from preprocess import load_and_clean_data

X, y, vectorizer = load_and_clean_data(r"C:\Academics 25-26\4th Sem\My Work\Projects\spam_email_classifier\ml\data\SMSSpamCollection")

print(X.shape)
print(y.value_counts())