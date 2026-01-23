# 📧 Spam Email Classifier – End-to-End ML & Django Project

An end-to-end Spam Email Classifier built using **Machine Learning (NLP)** and **Django**, featuring a clean UI, real-time predictions, and cloud deployment.

This project demonstrates the complete lifecycle of an ML application — from data preprocessing and model training to web integration and deployment.

---

### 🚀 Live Demo

**🔗 Deployed Application:** https://spam-email-classifier-c5lb.onrender.com/

---

### 📌 Project Overview

Spam detection is a classic text classification problem where the goal is to classify a message as either:

* **Spam** 🚨
* **Not Spam (Ham)** ✅

In this project, I built a full-scale ML system that:

* Learns patterns from real-world text data.
* Converts raw text into numerical features using NLP.
* Classifies unseen emails in real time via a web interface.

---

### 🧠 Machine Learning Workflow

**Dataset**

* SMS Spam Collection Dataset.
* Labeled as spam or ham.
* Imbalanced dataset handled carefully during training.

**Text Preprocessing**

1. Lowercasing
2. Punctuation removal
3. Tokenization
4. Stopword removal
5. **TF-IDF Vectorization**

**Model Training**

* **Algorithm:** Multinomial Naive Bayes.
* Train-test split with stratification.
* **Evaluation using:** Accuracy, Confusion Matrix, Precision, Recall, and F1-score.

**Model Persistence**

* Trained model and vectorizer saved using `pickle`.
* Ensures consistent preprocessing during prediction.

---

### 🧪 Technologies Used

| Category | Tools & Libraries |
| --- | --- |
| **Machine Learning & NLP** | Python, NumPy, Pandas, Scikit-learn, NLTK |
| **Backend** | Django, Gunicorn, WhiteNoise |
| **Frontend** | HTML, Bootstrap 5, Vanilla JavaScript |
| **Deployment** | Render (Cloud Hosting), GitHub (Version Control) |

---

### 🎨 Features

* **Real-time** spam prediction.
* Clean, responsive UI using **Bootstrap**.
* **Dark mode toggle** 🌙.
* Character counter for input text.
* Loading spinner for better UX.
* Example spam & ham inputs.
* Clear input button.
* Production-ready deployment.

---

### 🏗️ Project Structure

```text
spam-email-classifier/
│
├── ml/
│   ├── preprocess.py     # Text preprocessing pipeline
│   ├── train.py          # Model training & evaluation
│   ├── predict.py        # Prediction logic
│   └── model.pkl         # Saved ML model
│
├── webapp/
│   ├── manage.py
│   ├── detector/
│   │   ├── views.py
│   │   └── templates/
│   │       └── home.html
│   └── webapp/
│       └── settings.py
│
├── requirements.txt
├── render.yaml
└── README.md

```

---

### ⚙️ How to Run Locally

**1️⃣ Clone the Repository**

```bash
git clone https://github.com/your-username/spam-email-classifier.git
cd spam-email-classifier

```

**2️⃣ Create & Activate Virtual Environment**

```bash
python -m venv venv
venv\Scripts\activate   # Windows

```

**3️⃣ Install Dependencies**

```bash
pip install -r requirements.txt

```

**4️⃣ Run Django Server**

```bash
cd webapp
python manage.py runserver

```

Open: `http://127.0.0.1:8000/`

---

### 📊 Model Performance

* **Accuracy:** ~97%
* Strong precision & recall for spam detection.
* **Confusion matrix** used to evaluate false positives & false negatives.
* Accuracy alone was not relied upon due to class imbalance.

---

### 📚 What I Learned

* End-to-end ML pipeline design.
* Text preprocessing & NLP fundamentals.
* Handling imbalanced datasets.
* Integrating ML models with Django.
* Building user-friendly web interfaces.
* Deploying ML-powered web apps to cloud.
* Writing production-ready, modular code.

---

### 🔮 Future Improvements

* Use Logistic Regression / SVM for comparison.
* Add confidence score to predictions.
* Add user authentication.
* Store prediction history.
* Convert app into REST API.
* Dockerize the application.

---

### 👨‍💻 Author

**Akshat Raj** *BCA Student | Aspiring ML & AI Engineer*

🔗 **LinkedIn:** www.linkedin.com/in/akshat-sys01

🔗 **GitHub:** https://github.com/Akshat-sys01/spam-email-classifier.git

---

### ⭐ Acknowledgements

* UCI SMS Spam Collection Dataset.
* Scikit-learn & Django documentation.
