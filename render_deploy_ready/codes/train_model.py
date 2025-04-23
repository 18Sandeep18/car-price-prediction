import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
import os

# Load data
car_df = pd.read_csv('D:/Smartbriddge/SI-GuidedProject-615547-1700552397-main/codes/car_data.csv')

# Prepare data
X = car_df.iloc[:, [2, 3]]  # Age and EstimatedSalary
y = car_df.iloc[:, 4]       # Purchased

# Train-test split
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.25, random_state=1)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Create directory to save models
os.makedirs('model', exist_ok=True)

# Save model and scaler
with open('codes/model/model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('codes/model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully!")
