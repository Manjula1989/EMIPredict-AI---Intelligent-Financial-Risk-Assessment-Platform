import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# ✅ Load dataset
df = pd.read_csv("emi_prediction_dataset.csv", low_memory=False)
print(f"📊 Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ✅ Define target column
target_col = 'max_monthly_emi'

# ✅ Identify categorical columns automatically
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"🔹 Categorical columns detected: {categorical_cols}")

# ✅ Encode categorical columns and save LabelEncoders
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Convert to string to avoid dtype issues
    df[col] = df[col].astype(str)
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
print("✅ Categorical columns encoded and LabelEncoders saved")

# ✅ Split features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Ensure all numeric columns are numeric
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# Fill any NaN values with 0 (or you can choose median)
X = X.fillna(0)
y = y.fillna(0)

# ✅ Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"🔹 Training set: {X_train.shape}, Test set: {X_test.shape}")

# ✅ Train RandomForestRegressor
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
print("✅ RandomForestRegressor trained")

# Evaluate model
r2 = rf_model.score(X_test, y_test)
mse = np.mean((rf_model.predict(X_test) - y_test) ** 2)
print(f"✅ Model R²: {r2:.4f}, MSE: {mse:.4f}")

# ✅ Create 'models' folder if it doesn't exist
if not os.path.exists("models"):
    os.makedirs("models")

# ✅ Save model and label encoders
joblib.dump(rf_model, "models/best_regression_model.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")
print("💾 Model and LabelEncoders saved successfully!")
