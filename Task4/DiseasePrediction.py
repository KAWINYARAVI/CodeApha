import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_breast_cancer

# Load dataset (Breast Cancer Wisconsin dataset)
data = load_breast_cancer()
X = data.data  # Features
y = data.target  # Target (0: malignant, 1: benign)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Classification report (precision, recall, f1-score)
print(classification_report(y_test, y_pred))

# Example of predicting disease likelihood for new data
# Replace `new_data` with your own medical data
new_data = np.array([[14.5, 15.3, 91.38, 648.2, 0.104, 0.08, 0.07, 0.025, 0.175, 0.065, 0.245, 1.365, 1.683, 22.75, 0.0039, 0.0121, 0.02, 0.0078, 0.0151, 0.002, 17.42, 19.07, 111.6, 939.7, 0.137, 0.165, 0.17, 0.082, 0.297, 0.084]])

# Reshape for prediction (if necessary)
new_data = new_data.reshape(1, -1)

# Predict disease likelihood
prediction = model.predict(new_data)
if prediction[0] == 0:
    print("Predicted: Malignant (cancerous)")
else:
    print("Predicted: Benign (non-cancerous)")
