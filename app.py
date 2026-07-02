import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

# =============================
# Load Models and Vectorizer
# =============================
nb = pickle.load(open("model_nb.pkl", "rb"))
lr = pickle.load(open("model_lr.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# =============================
# Load Users
# =============================
users = pd.read_csv("users.csv")

st.set_page_config(page_title="Mental Health Stress Detection")

st.title("Mental Health Stress Detection System")
st.warning("⚠️ Educational purpose only – Not a medical diagnosis")

# =============================
# Login System
# =============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if ((users["username"] == username) &
            (users["password"] == password)).any():
            st.session_state.logged_in = True
            st.session_state.user = username
            st.success("Login successful")
        else:
            st.error("Invalid username or password")
    st.stop()

st.success(f"Welcome, {st.session_state.user}")

# =============================
# Model Selection (NEW)
# =============================
model_choice = st.selectbox(
    "Select Machine Learning Model",
    ["Naive Bayes", "Logistic Regression"]
)

# =============================
# Stress Prediction Section
# =============================
st.subheader("Stress Level Prediction")

text = st.text_area("Enter your thoughts or feelings")

# Consent Checkbox (NEW)
consent = st.checkbox(
    "I understand this system is for educational purposes only and not a medical diagnosis."
)

if st.button("Predict Stress"):
    if not consent:
        st.warning("Please provide consent to proceed.")
        st.stop()

    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        vec = vectorizer.transform([text])

        if model_choice == "Naive Bayes":
            pred = nb.predict(vec)[0]
            prob = nb.predict_proba(vec).max()
        else:
            pred = lr.predict(vec)[0]
            prob = lr.predict_proba(vec).max()

        explanation = {
            "Low": "You seem relaxed. Maintain a healthy routine.",
            "Medium": "You may be experiencing manageable stress.",
            "High": "High stress detected. Consider taking breaks or support."
        }

        # Output
        st.success(f"Stress Level: {pred}")

        # Confidence Coloring (NEW)
        confidence = round(prob * 100, 2)
        if confidence > 80:
            st.success(f"Confidence: {confidence}%")
        elif confidence >= 50:
            st.warning(f"Confidence: {confidence}%")
        else:
            st.error(f"Confidence: {confidence}%")

        st.write(explanation[pred])

        # =============================
        # Explainable AI (Feature 8)
        # =============================
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = vec.toarray()[0]

        top_indices = tfidf_scores.argsort()[-3:][::-1]
        important_words = [
            feature_names[i]
            for i in top_indices
            if tfidf_scores[i] > 0
        ]

        if important_words:
            st.write(
                "Key words influencing prediction:",
                ", ".join(important_words)
            )

        # =============================
        # History Logging
        # =============================
        row = {
            "user": st.session_state.user,
            "text": text,
            "prediction": pred,
            "confidence": confidence,
            "time": datetime.now()
        }

        try:
            history = pd.read_csv("history.csv")
        except:
            history = pd.DataFrame(columns=row.keys())

        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
        history.to_csv("history.csv", index=False)

# =============================
# Load History
# =============================
try:
    history = pd.read_csv("history.csv")
    history["time"] = pd.to_datetime(history["time"])
    user_hist = history[history["user"] == st.session_state.user]
except:
    user_hist = pd.DataFrame()

# =============================
# User Stress Summary
# =============================
st.subheader("Your Stress Summary")

if not user_hist.empty:
    st.write(user_hist["prediction"].value_counts())
else:
    st.info("No history available yet.")

# =============================
# Weekly Stress Summary (NEW)
# =============================
st.subheader("Weekly Stress Summary")

last_week = user_hist[
    user_hist["time"] >= pd.Timestamp.now() - pd.Timedelta(days=7)
]

if not last_week.empty:
    st.bar_chart(last_week["prediction"].value_counts())
else:
    st.info("No data for the last 7 days.")

# =============================
# Download Report
# =============================
if not user_hist.empty:
    csv = user_hist.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Stress Report (CSV)",
        csv,
        "stress_report.csv",
        "text/csv"
    )

# =============================
# Rule-Based Stress Alert
# =============================
if len(user_hist) >= 3:
    recent = user_hist.tail(3)
    if all(recent["prediction"] == "High"):
        st.error(
            "⚠️ Repeated high stress detected. "
            "Please consider taking breaks or seeking support."
        )

# =============================
# Data Deletion Option (NEW)
# =============================
st.subheader("Data Privacy")

if st.button("Delete My Stress History"):
    try:
        history = history[history["user"] != st.session_state.user]
        history.to_csv("history.csv", index=False)
        st.success("Your stress history has been deleted.")
    except:
        st.info("No data to delete.")

# =============================
# Admin Dashboard
# =============================
if st.session_state.user == "admin":
    st.subheader("Admin Dashboard")

    try:
        st.write("Total Predictions:", len(history))
        st.write("Overall Stress Distribution:")
        st.bar_chart(history["prediction"].value_counts())

        keywords = ["stress", "anxious", "tired", "pressure", "worried"]
        text_data = " ".join(history["text"].astype(str).str.lower())
        keyword_count = {k: text_data.count(k) for k in keywords}

        st.write("Common Stress Keywords:")
        st.write(keyword_count)
    except:
        st.info("No data available yet.")


