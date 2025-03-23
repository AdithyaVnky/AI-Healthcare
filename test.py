import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model

# Load test dataset
test_df = pd.read_csv("sample_test_patients.csv")

# Convert Date column to datetime
test_df['Date'] = pd.to_datetime(test_df['Date'])

# Sort by Patient_ID and Date
test_df = test_df.sort_values(by=['Patient_ID', 'Date'])

# Features for LSTM (wearable time-series data)
time_series_features = ['Heart_Rate', 'Sleep_Hours', 'Calories_Burnt', 'Activity_Level']

# Features for XGBoost (medical history)
medical_features = ['Age', 'Gender', 'BMI', 'Blood_Pressure', 'Total_Cholesterol', 
                    'LDL_Cholesterol', 'HDL_Cholesterol', 
                    'Heart_Disease_Family_History', 'Arthritis_Family_History', 'Diabetes_Family_History']

def prepare_lstm_test_data(df, patient_id_col, time_features, timesteps=30):
    X = []
    grouped = df.groupby(patient_id_col)
    
    for _, group in grouped:
        group = group.sort_values('Date')
        sequences = group[time_features].values
        
        if len(group) >= timesteps:
            X.append(sequences[-timesteps:])  # Use the last 30 days for each patient
    
    return np.array(X)

# Prepare LSTM input data
X_lstm_test = prepare_lstm_test_data(test_df, 'Patient_ID', time_series_features)

# Load trained LSTM model
lstm_model = load_model("lstm_model.h5")

# Extract LSTM predictions as features
lstm_test_features = lstm_model.predict(X_lstm_test).flatten()

# Extract medical history data (one row per patient)
test_df_unique = test_df.drop_duplicates(subset=['Patient_ID'])
medical_test_data = test_df_unique[medical_features].values

# Combine LSTM Features with Medical Data for XGBoost
X_test_combined = np.column_stack((medical_test_data, lstm_test_features))

# Load trained XGBoost model
xgb_model = joblib.load("xgb_model.pkl")

# Make Predictions
predictions = xgb_model.predict(X_test_combined)

# Display Predictions
for i, pred in enumerate(predictions):
    print(f"Patient {i+1} - Predicted Disease Risk: {pred}")
