import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('student_performance_model.pkl')

# Streamlit UI
st.title("Student Performance Predictor 🎓")
st.write("Predict marks based on attendance and study hours.")

# Inputs
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=75)
hours = st.number_input("Study Hours per day", min_value=0.0, max_value=12.0, value=4.0, step=0.5)

# Predict button
if st.button("Predict Marks"):
    input_df = pd.DataFrame({'Attendance': [attendance], 'Hours': [hours]})
    predicted_marks = model.predict(input_df)[0]
    st.success(f"Predicted Marks: {predicted_marks:.2f}")