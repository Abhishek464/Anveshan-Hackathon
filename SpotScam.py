import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForigin mainestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

import streamlit as st

# ---------- Preprocessing Class ----------
class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.fillna("").str.lower().str.replace("[^a-zA-Z0-9 ]", "", regex=True)

# ---------- Model Training ----------
def train_model():
    df = pd.read_csv("/TrainingData.csv")  # Dataset with 'fraudulent' column
    df = df.dropna(subset=["title", "description"])
    df["text"] = df["title"] + " " + df["description"]

    X = df["text"]
    y = df["fraudulent"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("cleaner", TextCleaner()),
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print("F1 Score:", f1_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return pipeline

# ---------- Dashboard ----------
def run_dashboard(model):
    st.title("Fraudulent Job Post Detection")

    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        if not {"title", "description"}.issubset(data.columns):
            st.error("CSV must contain at least 'title' and 'description' columns.")
            return

        data["text"] = data["title"].fillna("") + " " + data["description"].fillna("")
        probs = model.predict_proba(data["text"])[::, 1]
        preds = model.predict(data["text"])

        data["fraud_probability"] = probs
        data["prediction"] = preds

        st.subheader("Prediction Table")
        st.dataframe(data[["title", "description", "fraud_probability", "prediction"]])

        st.subheader("Fraud Probability Histogram")
        fig, ax = plt.subplots()
        sns.histplot(data["fraud_probability"], bins=20, kde=False, ax=ax)
        st.pyplot(fig)

        st.subheader("Fraud vs Genuine Pie Chart")
        fig2, ax2 = plt.subplots()
        labels = ["Genuine", "Fraudulent"]
        sizes = [sum(data["prediction"] == 0), sum(data["prediction"] == 1)]
        ax2.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        st.pyplot(fig2)

        st.subheader("Top 10 Most Suspicious Listings")
        top10 = data.sort_values("fraud_probability", ascending=False).head(10)
        st.table(top10[["title", "description", "fraud_probability"]])

# ---------- Main ----------
if __name__ == "__main__":
    model = train_model()
    run_dashboard(model)