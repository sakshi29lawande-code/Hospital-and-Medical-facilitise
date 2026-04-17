import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="AI Hospital System", layout="wide")

st.title("🏥 AI Hospital & Disease Prediction System")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("disease_data.csv")

df = load_data()

X = df.drop("disease", axis=1)
y = df["disease"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# -----------------------------
# SIDEBAR - BMI CALCULATOR
# -----------------------------
st.sidebar.header("📊 BMI Calculator")

height = st.sidebar.number_input("Height (cm)", 100, 220)
weight = st.sidebar.number_input("Weight (kg)", 30, 150)

bmi = weight / ((height/100) ** 2)

if st.sidebar.button("Calculate BMI"):
    st.sidebar.write(f"💡 BMI: {round(bmi,2)}")

    if bmi < 18.5:
        st.sidebar.warning("Underweight")
    elif bmi < 25:
        st.sidebar.success("Normal")
    else:
        st.sidebar.error("Overweight")

# -----------------------------
# USER INPUT
# -----------------------------
st.header("🧾 Select Symptoms")

def user_input(label):
    return st.checkbox(label)

symptoms = [
    "fever","cough","headache","fatigue","sore_throat",
    "runny_nose","chest_pain","short_breath","vomiting","body_pain"
]

input_data = {sym: user_input(sym.replace("_"," ").title()) for sym in symptoms}
input_df = pd.DataFrame([input_data])

# -----------------------------
# Doctor + Medicine Database
# -----------------------------
doctor_db = {
    "Flu": "General Physician",
    "Migraine": "Neurologist",
    "Common Cold": "ENT Specialist",
    "Heart Problem": "Cardiologist",
    "Food Poisoning": "Gastroenterologist",
    "Viral Fever": "General Physician"
}

medicine_db = {
    "Flu": ["Paracetamol", "Rest", "Fluids"],
    "Migraine": ["Ibuprofen", "Aspirin"],
    "Common Cold": ["Cetirizine", "Cough Syrup"],
    "Heart Problem": ["Emergency Care Required"],
    "Food Poisoning": ["ORS", "Domperidone"],
    "Viral Fever": ["Paracetamol"]
}

diet_db = {
    "Flu": "Drink warm fluids, soups, fruits",
    "Migraine": "Avoid caffeine, eat light meals",
    "Common Cold": "Hot water, ginger tea",
    "Heart Problem": "Low salt, avoid oily food",
    "Food Poisoning": "ORS, banana, rice",
    "Viral Fever": "Hydration, light diet"
}

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict"):

    prediction = model.predict(input_df)[0]

    # Probability
    prob = model.predict_proba(input_df).max()

    st.subheader("🧾 Result")
    st.success(f"🦠 Disease: {prediction}")
    st.write(f"📊 Confidence: {round(prob*100,2)}%")

    # Doctor
    doctor = doctor_db.get(prediction, "General Physician")
    st.info(f"👨‍⚕️ Recommended Doctor: {doctor}")

    # Medicines
    st.subheader("💊 Medicines")
    meds = medicine_db.get(prediction, ["Consult doctor"])
    for m in meds:
        st.write(f"✔ {m}")

    # Diet
    st.subheader("🥗 Diet Suggestion")
    st.write(diet_db.get(prediction, "Healthy diet"))

    # Severity
    score = sum(input_data.values())

    if score >= 6:
        severity = "High"
        advice = "⚠️ Visit hospital immediately!"
    elif score >= 3:
        severity = "Medium"
        advice = "🩺 Consult doctor soon."
    else:
        severity = "Low"
        advice = "✅ Take rest and monitor symptoms."

    st.subheader("📊 Severity Level")
    st.write(f"{severity}")
    st.warning(advice)

    # Health Score
    health_score = 100 - (score * 10)

    st.subheader("📊 Health Score")
    st.progress(max(0, health_score)/100)
    st.write(f"{health_score}/100")

    # -----------------------------
    # Download Report
    # -----------------------------
    report = f"""
    Disease: {prediction}
    Confidence: {round(prob*100,2)}%
    Doctor: {doctor}
    Severity: {severity}
    Health Score: {health_score}
    """

    st.download_button("📄 Download Report", report, file_name="health_report.txt")

    st.caption("⚠️ This is not a real medical diagnosis. Please consult a doctor.")