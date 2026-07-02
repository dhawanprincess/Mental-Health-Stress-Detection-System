import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("dataset.csv")

X = df["text"]
y = df["stress_level"]

vectorizer = TfidfVectorizer(stop_words="english")
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Model 1: Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)

# Model 2: Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

pickle.dump(nb, open("model_nb.pkl", "wb"))
pickle.dump(lr, open("model_lr.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Models trained successfully")
    
  