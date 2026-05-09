import pandas as pd
import joblib
import shap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from preprocessing import DataPreprocessor
import os

def train_and_evaluate():
    print("Loading data...")
    df = pd.read_csv('../data/Diabetes_detectionDS1.csv')
    
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess_train(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training XGBoost with Hyperparameter Tuning...")
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    param_grid = {
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1]
    }
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='roc_auc', verbose=1)
    grid_search.fit(X_train, y_train)
    best_xgb = grid_search.best_estimator_
    
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    
    print("Building Ensemble Voting Classifier...")
    ensemble = VotingClassifier(estimators=[
        ('xgb', best_xgb), ('rf', rf_model)
    ], voting='soft')
    ensemble.fit(X_train, y_train)
    
    # Evaluation
    preds = ensemble.predict(X_test)
    probs = ensemble.predict_proba(X_test)[:, 1]
    
    print("\n--- MODEL EVALUATION ---")
    print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, probs):.4f}")
    print(classification_report(y_test, preds))
    
    # Explainable AI (SHAP based on XGBoost to be fast)
    print("Generating SHAP Explainer...")
    explainer = shap.TreeExplainer(best_xgb)
    
    # Save Models & Artifacts
    print("Saving Models to /models...")
    preprocessor.save_artifacts('../models/')
    joblib.dump(ensemble, '../models/ensemble_model.pkl')
    joblib.dump(best_xgb, '../models/xgb_model.pkl') # Saved for SHAP inference
    joblib.dump(explainer, '../models/shap_explainer.pkl')
    print("Training complete and models saved successfully.")

if __name__ == "__main__":
    train_and_evaluate()