import streamlit as st
import pandas as pd
from datetime import datetime
import sqlite3
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import hashlib

# --- Session State for Login ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# --- Login Page ---
def login():
    st.title("🔐 Doctor Login")
    password = st.text_input("Enter Password", type="password")
    if st.button("Login"):
        if password == "doctor123":
            st.session_state.authenticated = True
            st.success("✅ Logged in!")
            st.rerun()
        else:
            st.error("❌ Incorrect password")

if not st.session_state.authenticated:
    login()
    st.stop()

# --- Navigation Sidebar ---
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict", "Patients", "Logout"])
classifier_option = st.sidebar.selectbox("🤖 Choose Classifier", ["Logistic Regression", "Random Forest", "XGBoost"])
use_encryption = st.sidebar.checkbox("🔒 Encrypt Patient Name")
use_sqlite = st.sidebar.checkbox("🗃️ Use SQLite (instead of CSV)")

# --- Load Dataset & Train Model ---
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

if classifier_option == "Logistic Regression":
    model = LogisticRegression(max_iter=10000)
elif classifier_option == "Random Forest":
    model = RandomForestClassifier()
else:
    model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)

model.fit(X, y)

# --- SQLite Setup ---
if use_sqlite:
    conn = sqlite3.connect("patients.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            name TEXT,
            age INTEGER,
            gender TEXT,
            prediction TEXT,
            confidence TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()

# --- Pages ---
if page == "Home":
    st.title("🏠 Welcome to the Breast Cancer Prediction App")
    st.markdown("""
    This app helps medical professionals predict the likelihood of breast cancer (Benign or Malignant) using machine learning.

    **Features:**
    - Input patient details and medical test results
    - Predict cancer with high accuracy
    - Store and review patient history
    - Download or search records
    - Choose from Logistic Regression, Random Forest, or XGBoost
    - SQLite or CSV support
    - Login security and encrypted names

    👉 Use the sidebar to navigate.
    """)

elif page == "Predict":
    st.title("🔍 Cancer Prediction")

    with st.form("prediction_form"):
        st.subheader("Patient Information")
        name = st.text_input("Name")
        age = st.number_input("Age", min_value=1, max_value=120)
        gender = st.radio("Gender", ["Male", "Female", "Other"])

        st.subheader("Medical Test Inputs")
        user_input = {}
        for feature in data.feature_names:
            user_input[feature] = st.number_input(
                feature, min_value=0.0, format="%.4f", help=f"Input for {feature}"
            )

        submitted = st.form_submit_button("Predict")

    if submitted:
        if not name or age <= 0:
            st.warning("⚠️ Please enter valid name and age.")
        else:
            input_df = pd.DataFrame([user_input])
            prediction = model.predict(input_df)[0]
            confidence = model.predict_proba(input_df)[0][prediction]
            result = "Benign" if prediction == 1 else "Malignant"

            st.success(f"Prediction: {result}")
            st.info(f"Confidence: {confidence * 100:.2f}%")

            stored_name = hashlib.sha256(name.encode()).hexdigest() if use_encryption else name

            if use_sqlite:
                cursor.execute('''
                    INSERT INTO patients (name, age, gender, prediction, confidence, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (stored_name, age, gender, result, f"{confidence*100:.2f}%",
                      datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                conn.commit()
                st.success("✅ Data saved to database!")
            else:
                record = {
                    "Name": stored_name,
                    "Age": age,
                    "Gender": gender,
                    "Prediction": result,
                    "Confidence": f"{confidence * 100:.2f}%",
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                record.update(user_input)

                df = pd.DataFrame([record])
                try:
                    existing = pd.read_csv("patients.csv")
                    df = pd.concat([existing, df], ignore_index=True)
                except FileNotFoundError:
                    pass

                df.to_csv("patients.csv", index=False)
                st.success("✅ Data saved to CSV!")

elif page == "Patients":
    st.title("📋 Patient Records")

    if use_sqlite:
        cursor.execute("SELECT * FROM patients")
        rows = cursor.fetchall()
        db_df = pd.DataFrame(rows, columns=["Name", "Age", "Gender", "Prediction", "Confidence", "Timestamp"])
        st.dataframe(db_df)

        st.download_button("⬇️ Download SQLite Data", db_df.to_csv(index=False), "patients.csv", "text/csv")
    else:
        try:
            patient_data = pd.read_csv("patients.csv")
            search = st.text_input("🔍 Search by Name")
            if search:
                filtered = patient_data[patient_data["Name"].str.contains(search, case=False, na=False)]
                st.dataframe(filtered)
            else:
                st.dataframe(patient_data)

            if st.button("📊 Show Prediction Summary"):
                summary = patient_data["Prediction"].value_counts()
                st.bar_chart(summary)

            st.download_button("⬇️ Download CSV", patient_data.to_csv(index=False), "patients.csv", "text/csv")

        except FileNotFoundError:
            st.warning("No CSV data found.")

elif page == "Logout":
    st.session_state.authenticated = False
    st.success("✅ Logged out.")
    st.experimental_rerun()

# --- Mobile Responsive CSS ---
st.markdown("""
<style>
body {
    font-size: 16px;
}
section.main > div {
    max-width: 95%;
    margin: auto;
}
</style>
""", unsafe_allow_html=True)
