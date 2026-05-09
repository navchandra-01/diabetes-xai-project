import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import os

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def preprocess_train(self, df):
        df = df.copy()
        
        # 1. Handle Missing Values
        df['Alcohol_Intake'] = df['Alcohol_Intake'].fillna('None')
        
        # 2. Fix inconsistent data types
        df['Polycystic_Ovary_Syndrome'] = df['Polycystic_Ovary_Syndrome'].replace({'0': 'No', 0: 'No'})
        
        # 3. Feature Engineering
        df['BMI_Cat'] = pd.cut(df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], 
                               labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        df['Glucose_Ratio'] = df['Postprandial_Blood_Sugar'] / (df['Fasting_Blood_Sugar'] + 1e-5)
        
        # 4. Target Encoding
        le_target = LabelEncoder()
        df['Diabetes_Status'] = le_target.fit_transform(df['Diabetes_Status'])
        self.label_encoders['Diabetes_Status'] = le_target
        
        y = df['Diabetes_Status']
        X = df.drop('Diabetes_Status', axis=1)
        
        # 5. One-Hot Encoding
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        self.final_columns = X.columns
        
        # 6. Scaling
        num_cols = X.select_dtypes(exclude=['uint8', 'bool']).columns
        X[num_cols] = self.scaler.fit_transform(X[num_cols])
        self.num_cols = num_cols
        
        return X, y

    def preprocess_inference(self, df):
        """Used by the API for real-time predictions"""
        df = df.copy()
        df['Alcohol_Intake'] = df.get('Alcohol_Intake', 'None')
        df['Polycystic_Ovary_Syndrome'] = str(df.get('Polycystic_Ovary_Syndrome', 'No')).replace('0', 'No')
        
        df['BMI_Cat'] = pd.cut(df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        df['Glucose_Ratio'] = df['Postprandial_Blood_Sugar'] / (df['Fasting_Blood_Sugar'] + 1e-5)
        
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        
        # Align with training columns
        for col in self.final_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[self.final_columns]
        
        df[self.num_cols] = self.scaler.transform(df[self.num_cols])
        return df

    def save_artifacts(self, path="models/"):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(path, "scaler.pkl"))
        joblib.dump(self.final_columns, os.path.join(path, "columns.pkl"))
        joblib.dump(self, os.path.join(path, "preprocessor.pkl"))