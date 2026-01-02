import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("heart_disease_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("❤️ Heart Disease Prediction System")
st.warning("⚠️ This tool is for educational purposes only.")

st.subheader("Enter Patient Details")

age = st.number_input("Age", 18, 100)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
chest_pain = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
bp = st.number_input("Resting Blood Pressure", 80, 200)
chol = st.number_input("Cholesterol Level", 100, 400)
fbs = st.selectbox("FBS over 120 mg/dl (0 = No, 1 = Yes)", [0, 1])
ekg = st.selectbox("EKG Results (0–2)", [0, 1, 2])
max_hr = st.number_input("Maximum Heart Rate", 60, 220)
exercise_angina = st.selectbox("Exercise Induced Angina (0 = No, 1 = Yes)", [0, 1])
st_depression = st.number_input("ST Depression", 0.0, 10.0)
slope = st.selectbox("Slope of ST (0–2)", [0, 1, 2])
vessels = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thallium = st.selectbox("Thallium Test Result (0–3)", [0, 1, 2, 3])

if st.button("Predict Risk"):
    input_data = np.array([[age, sex, chest_pain, bp, chol, fbs, ekg,
                            max_hr, exercise_angina, st_depression,
                            slope, vessels, thallium]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ High risk of heart disease detected")
    else:
        st.success("✅ Low risk of heart disease detected")
