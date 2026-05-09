import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Diabetes Risk AI", layout="wide")

st.title("🩺 Explainable AI-Driven Diabetes Risk Assessment")
st.markdown("Enter patient parameters to assess diabetes risk in real-time with AI-generated explainability.")

# Sidebar Inputs
st.sidebar.header("Patient Vitals")
age = st.sidebar.slider("Age", 18, 100, 45)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
bmi = st.sidebar.number_input("BMI", 15.0, 50.0, 25.0)
fbs = st.sidebar.number_input("Fasting Blood Sugar", 70.0, 300.0, 100.0)
hba1c = st.sidebar.number_input("HBA1C (%)", 4.0, 15.0, 5.5)
family_hist = st.sidebar.selectbox("Family History of Diabetes", ["Yes", "No"])
hypertension = st.sidebar.selectbox("Hypertension", ["Yes", "No"])

# For brevity, setting defaults for remaining features
patient_data = {
    "Age": age, "Gender": gender, "BMI": bmi, "Family_History": family_hist,
    "Physical_Activity": "Medium", "Smoking_Status": "Never", "Alcohol_Intake": "None",
    "Stress_Level": "Medium", "Hypertension": hypertension, "Cholesterol_Level": 180.0,
    "Fasting_Blood_Sugar": fbs, "Postprandial_Blood_Sugar": 140.0, "HBA1C": hba1c,
    "Heart_Rate": 72, "Waist_Hip_Ratio": 0.85, "Pregnancies": 0,
    "Polycystic_Ovary_Syndrome": "No", "Glucose_Tolerance_Test_Result": 120.0,
    "Vitamin_D_Level": 25.0, "C_Protein_Level": 3.0, "Thyroid_Condition": "No"
}

if st.button("Predict Diabetes Risk 🔍"):
    with st.spinner('Analyzing vitals with Ensemble AI...'):
        # In a real setup, requests.post to FastAPI: http://localhost:8000/predict
        # For demo purposes, assuming API response mapping:
        response = requests.post("http://127.0.0.1:8000/predict", json=patient_data)
        
        if response.status_code == 200:
            result = response.json()
            risk = result['risk_probability'] * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Risk Probability")
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = risk,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Diabetes Risk Score"},
                    gauge = {'axis': {'range': [None, 100]},
                             'bar': {'color': "darkred" if risk > 50 else "green"},
                             'steps': [
                                 {'range': [0, 30], 'color': "lightgreen"},
                                 {'range': [30, 70], 'color': "yellow"},
                                 {'range': [70, 100], 'color': "salmon"}]}
                ))
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                st.subheader("Explainable AI Insights (SHAP)")
                st.write(f"**Prediction Outcome:** {result['prediction']}")
                st.markdown("### Top Driving Factors:")
                for factor in result['top_risk_factors']:
                    impact = "Increased" if factor['contribution'] > 0 else "Decreased"
                    st.write(f"- **{factor['feature']}** {impact} the risk score by {abs(factor['contribution']):.2f}")