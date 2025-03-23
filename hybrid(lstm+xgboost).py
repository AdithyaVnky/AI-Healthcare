import pandas as pd
import numpy as np
import tensorflow as tf
import joblib  # Added joblib for saving/loading XGBoost model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Load the hybrid dataset
df = pd.read_csv("first_data.csv")

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort by Patient_ID and Date
df = df.sort_values(by=['Patient_ID', 'Date'])

# Features for LSTM (time-series data from wearables)
time_series_features = ['Heart_Rate', 'Sleep_Hours', 'Calories_Burnt', 'Activity_Level']

# Features for XGBoost (medical history)
medical_features = ['Age', 'Gender', 'BMI', 'Blood_Pressure', 'Total_Cholesterol', 
                    'LDL_Cholesterol', 'HDL_Cholesterol',
                    'Heart_Disease_Family_History', 'Arthritis_Family_History', 'Diabetes_Family_History']

target_column = 'Diabetes'  # Example target (can be changed)

# Prepare LSTM data (reshape to 3D: samples, timesteps, features)
def prepare_lstm_data(df, patient_id_col, time_features, target_col, timesteps=7):
    X, y = [], []
    grouped = df.groupby(patient_id_col)
    
    for _, group in grouped:
        group = group.sort_values('Date')
        sequences = group[time_features].values
        targets = group[target_col].values
        
        for i in range(len(group) - timesteps):
            X.append(sequences[i:i+timesteps])
            y.append(targets[i+timesteps])
            
    return np.array(X), np.array(y)

X_lstm, y_lstm = prepare_lstm_data(df, 'Patient_ID', time_series_features, target_column)

# Split LSTM data into training and testing sets
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

# Build LSTM Model
lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train LSTM Model
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32, validation_data=(X_test_lstm, y_test_lstm))

# Save LSTM Model
lstm_model.save("lstm_model.h5")
print("✅ LSTM model saved as lstm_model.h5")

# Extract LSTM Features (Predictions)
lstm_train_features = lstm_model.predict(X_train_lstm).flatten()
lstm_test_features = lstm_model.predict(X_test_lstm).flatten()

# Prepare XGBoost Data
medical_data = df.drop_duplicates(subset=['Patient_ID'])  # Keep one row per patient
X_medical = medical_data[medical_features]
y_medical = medical_data[target_column]

# Split medical data for XGBoost
X_train_med, X_test_med, y_train_med, y_test_med = train_test_split(X_medical, y_medical, test_size=0.2, random_state=42)

# Ensure LSTM Features Match XGBoost Data
min_len = min(len(X_train_med), len(lstm_train_features))
X_train_combined = np.column_stack((X_train_med[:min_len], lstm_train_features[:min_len]))

min_len_test = min(len(X_test_med), len(lstm_test_features))
X_test_combined = np.column_stack((X_test_med[:min_len_test], lstm_test_features[:min_len_test]))

# Train XGBoost Model
xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_combined, y_train_med[:min_len])

# Predict and Evaluate
predictions = xgb_model.predict(X_test_combined)
accuracy = accuracy_score(y_test_med[:min_len_test], predictions)
print(f'✅ Hybrid Model Accuracy: {accuracy:.4f}')

# Save XGBoost Model using Joblib
joblib.dump(xgb_model, "xgb_model.pkl")
print("✅ XGBoost model saved as xgb_model.pkl using Joblib")
