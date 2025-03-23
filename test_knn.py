import pandas as pd
import numpy as np
import joblib

# Load test data
test_data_path = "test_data01.csv"
test_df = pd.read_csv(test_data_path)

# Load the trained scaler
scaler = joblib.load("scaler.pkl")

# Features used in training
features = ['Age', 'Gender', 'BMI', 'Smoking_Status', 'Alcohol_Consumption', 'Physical_Activity', 
            'Blood_Pressure', 'Heart_Rate', 'Fasting_Blood_Glucose', 'HbA1c', 'Total_Cholesterol',
            'LDL_Cholesterol', 'HDL_Cholesterol', 'C_Reactive_Protein', 'Heart_Disease_Family_History',
            'Arthritis_Family_History', 'Diabetes_Family_History']

# Scale test data properly
X_test_scaled = scaler.transform(test_df[features])

# Load trained models and predict for all diseases
targets = ['Heart_Disease', 'Arthritis', 'Diabetes']
predictions = {}

for target in targets:
    model_filename = f"knn_model_{target.lower()}.pkl"
    knn_model = joblib.load(model_filename)
    
    # Predict using trained model
    predictions[target] = knn_model.predict(X_test_scaled)

# Convert predictions to DataFrame
predictions_df = pd.DataFrame(predictions)
print("Predictions for Test Data:\n", predictions_df)
