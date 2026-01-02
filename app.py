import streamlit as st
import pickle
import numpy as np

# Load saved model and scaler
model = pickle.load(open("heart_disease_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("❤️ Heart Disease Prediction System")
st.write("This application predicts heart disease risk using machine learning.")
st.warning("⚠️ This tool is for educational purposes only.")

st.subheader("Enter Patient Details")

age = st.number_input("Age", min_value=18, max_value=100)
sex = st.selectbox("Sex", ["Female", "Male"])
bp = st.number_input("Resting Blood Pressure", min_value=80, max_value=200)
chol = st.number_input("Cholesterol Level", min_value=100, max_value=400)
max_hr = st.number_input("Maximum Heart Rate", min_value=60, max_value=220)

if st.button("Predict Risk"):
    sex_value = 1 if sex == "Male" else 0

    input_data = np.array([[age, sex_value, bp, chol, max_hr]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ High risk of heart disease detected")
    else:
        st.success("✅ Low risk of heart disease detected")
