from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import shap

app = FastAPI(title="Explainable Diabetes Prediction API")

# Load artifacts
model = joblib.load('models/ensemble_model.pkl')
xgb_model = joblib.load('models/xgb_model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')
explainer = joblib.load('models/shap_explainer.pkl')

class PatientData(BaseModel):
    Age: int
    Gender: str
    BMI: float
    Family_History: str
    Physical_Activity: str
    Smoking_Status: str
    Alcohol_Intake: str
    Stress_Level: str
    Hypertension: str
    Cholesterol_Level: float
    Fasting_Blood_Sugar: float
    Postprandial_Blood_Sugar: float
    HBA1C: float
    Heart_Rate: int
    Waist_Hip_Ratio: float
    Pregnancies: int
    Polycystic_Ovary_Syndrome: str
    Glucose_Tolerance_Test_Result: float
    Vitamin_D_Level: float
    C_Protein_Level: float
    Thyroid_Condition: str

@app.post("/predict")
def predict_diabetes(patient: PatientData):
    # Convert to DataFrame
    df = pd.DataFrame([patient.dict()])
    
    # Preprocess
    X_processed = preprocessor.preprocess_inference(df)
    
    # Predict
    prob = model.predict_proba(X_processed)[0][1]
    prediction = int(prob > 0.5)
    
    # SHAP Explanability
    shap_values = explainer.shap_values(X_processed)
    
    # Top 3 contributing features
    feature_names = X_processed.columns
    contributions = list(zip(feature_names, shap_values[0]))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top_features = [{"feature": k, "contribution": float(v)} for k, v in contributions[:3]]
    
    return {
        "risk_probability": float(prob),
        "prediction": "Diabetic" if prediction == 1 else "Non-Diabetic",
        "top_risk_factors": top_features
    }