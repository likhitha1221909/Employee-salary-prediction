import streamlit as st
import pandas as pd
import joblib

# ✅ Load model and encoders
model = joblib.load("salary_model.pkl")
le_gender = joblib.load("le_gender.pkl")
le_edu = joblib.load("le_edu.pkl")
le_job = joblib.load("le_job.pkl")

# ✅ App Title
st.title("💼 Salary Prediction App")

# ✅ Input form
gender = st.selectbox("Gender", le_gender.classes_)
edu = st.selectbox("Education Level", le_edu.classes_)
job = st.selectbox("Job Title", le_job.classes_)
age = st.slider("Age", 18, 65, 30)
exp = st.slider("Years of Experience", 0, 40, 5)

# ✅ Create DataFrame
input_data = pd.DataFrame({
    "Gender": [le_gender.transform([gender])[0]],
    "Education Level": [le_edu.transform([edu])[0]],
    "Job Title": [le_job.transform([job])[0]],
    "Age": [age],
    "Years of Experience": [exp]
})

# ✅ Predict
if st.button("Predict Salary"):
    prediction = model.predict(input_data)
    st.success(f"💰 Predicted Salary: ₹ {prediction[0]:,.2f}")
