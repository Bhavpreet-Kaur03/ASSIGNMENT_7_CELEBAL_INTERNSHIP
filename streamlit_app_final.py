#!/usr/bin/env python
# coding: utf-8

# In[25]:


# app.py
import streamlit as st
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("💓 Heart Disease Prediction App")

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    data_path = "heart.csv"
    if not os.path.exists(data_path):
        st.error("❌ Dataset not found. Please make sure 'heart.csv' is in the same folder.")
        return None
    df = pd.read_csv(data_path)
    return df

df = load_data()

if df is not None:
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    st.markdown("""
    #### ℹ️ Categorical Feature Information:
    - `Gender`: Male = 1, Female = 0
    - `Smoking`: Yes = 1, No = 0
    - `Alcohol Intake`: Yes = 1, No = 0
    """)

    # ---------- PREPROCESS ----------
    target_col = 'Heart Disease'  # check your actual column name
    if target_col not in df.columns:
        st.error(f"❌ Target column '{target_col}' not found in dataset.")
        st.stop()

    # Encode categorical variables
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # ---------- TRAIN ----------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Save model
    with open("model_f.pkl", "wb") as f:
        pickle.dump(model, f)

    # ---------- EVALUATE ----------
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.subheader("📊 Model Performance")
    st.success(f"✅ Accuracy: {acc:.2f}")
    st.text("🧾 Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # ---------- PREDICTION ----------
    st.subheader("🩺 Predict Heart Disease")
    user_input = []
    for col in X.columns:
        val = st.number_input(f"{col}", value=float(df[col].mean()))
        user_input.append(val)

    if st.button("Predict"):
        with open("model.pkl", "rb") as f:
            clf = pickle.load(f)
        proba = clf.predict_proba([user_input])[0]
        pred = np.argmax(proba)
        confidence = np.max(proba) * 100

        if pred == 1:
            st.error(f"⚠️ High risk of Heart Disease. Confidence: {confidence:.2f}%")
        else:
            st.success(f"✅ Low risk of Heart Disease. Confidence: {confidence:.2f}%")


# In[ ]:




