import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE  
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data_path = "chronic_diseases_dataset_final.csv"
df = pd.read_csv(data_path)

# Features and targets
features = ['Age', 'Gender', 'BMI', 'Smoking_Status', 'Alcohol_Consumption', 'Physical_Activity', 
            'Blood_Pressure', 'Heart_Rate', 'Fasting_Blood_Glucose', 'HbA1c', 'Total_Cholesterol',
            'LDL_Cholesterol', 'HDL_Cholesterol', 'C_Reactive_Protein', 'Heart_Disease_Family_History',
            'Arthritis_Family_History', 'Diabetes_Family_History']

targets = ['Heart_Disease', 'Arthritis', 'Diabetes']

# Scale features
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Train and save models for each disease
for target in targets:
    y = df[target]

    # 🔹 Print original class distribution
    print(f"Original Class Distribution for {target}:")
    print(y.value_counts(), "\n")

    # 🔹 Apply SMOTE to fix class imbalance
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    # 🔹 Print new class distribution after SMOTE
    print(f"Class Distribution for {target} after SMOTE:")
    print(pd.Series(y_resampled).value_counts(), "\n")

    # 🔹 Split dataset (random seed for better variation)
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=np.random.randint(1, 1000)
    )

    # 🔹 Train KNN model with improved hyperparameters
    knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan', weights='distance')
    knn.fit(X_train, y_train)

    # 🔹 Save the trained model
    model_filename = f"knn_model_{target.lower()}.pkl"
    joblib.dump(knn, model_filename)
    print(f"Model saved: {model_filename}")

    # 🔹 Evaluate the model
    y_pred = knn.predict(X_test)
    print(f'KNN Model Accuracy for {target}: {accuracy_score(y_test, y_pred):.2f}')
    print(f'Classification Report for {target}:\n', classification_report(y_test, y_pred))
