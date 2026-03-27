# Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('heart_attack_youngsters_india.csv')

# Converting the Categorical Features into Numerical
# For Ordinal Type of Data use 'Integer Encoding'
# Features to encode with Integer Encoding
integer_encoded_features = ['Physical Activity Level', 'Stress Level', 'Heart Attack Likelihood']

# Importing LabelEncoder (for Integer Encoding)
from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
encoder = LabelEncoder()

# Apply Integer Encoding to specific features
for col in integer_encoded_features:
    df[col] = encoder.fit_transform(df[col])

# Converting the Categorical Features into Numerical
# For all other categorical data use 'One Hot Encoding'
# Features to encode with One hot Encoding
one_hot_encoded_features = [
    'Gender', 'Region', 'Smoking Status','Alcohol Consumption', 'Diet Type','ECG Results',
    'Chest Pain Type','Diabetes', 'Hypertension', 'Exercise Induced Angina']


# Apply one hot encoding using pandas get_dummies
# Apply one-hot encoding using pandas get_dummies
df = pd.get_dummies(data=df, columns=one_hot_encoded_features, drop_first=True).astype(int)

# Features (X)
X = df.drop('Heart Attack Likelihood', axis=1)
# Target variable (y)
y = df['Heart Attack Likelihood']

# import the library
from sklearn.model_selection import train_test_split

# Split the data
# 30% of Training Data with a random seed number 101
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# import the library
from sklearn.preprocessing import StandardScaler

# Creating an Instance for the StandardScaler
scaler = StandardScaler()

# Scaling and training with X_train
scaled_X_train = scaler.fit_transform(X_train)
# Scaling X_test (Not training)
scaled_X_test = scaler.transform(X_test)

# import the model
from sklearn.linear_model import LogisticRegression

# Creating an Instance for the model
log_model = LogisticRegression(max_iter=1000)

# Train the model on the training data
log_model.fit(scaled_X_train, y_train)

# Predict the target values for the test set
y_pred = log_model.predict(scaled_X_test)

# Predict probabilities for the test set
y_prob = log_model.predict_proba(scaled_X_test)

# Import the classification metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Accuracy Score
accuracy_score(y_test,y_pred)

# Confusion Matrix
cm = confusion_matrix(y_test,y_pred)
cm

# Classification Report
print(classification_report(y_test,y_pred))


## New Prediction
# Example new patient data (modify values as needed)

new_sample = {
    'Age': 28,
    'Cholesterol Level': 190,
    'BMI': 24,
    'Physical Activity Level': 'Medium',
    'Stress Level': 'High',
    'Gender': 'Male',
    'Region': 'Urban',
    'Smoking Status': 'No',
    'Alcohol Consumption': 'Occasional',
    'Diet Type': 'Balanced',
    'ECG Results': 'Normal',
    'Chest Pain Type': 'Non-anginal',
    'Diabetes': 'No',
    'Hypertension': 'No',
    'Exercise Induced Angina': 'No'
}


new_df = pd.DataFrame([new_sample])

for col in ['Physical Activity Level', 'Stress Level']:
    new_df[col] = encoder.fit_transform(new_df[col])

new_df = pd.get_dummies(new_df)

new_df = new_df.reindex(columns=X.columns, fill_value=0)

new_scaled = scaler.transform(new_df)

prediction = log_model.predict(new_scaled)
probability = log_model.predict_proba(new_scaled)

print("\nManual Input Prediction:")
print("Predicted Class:", prediction[0])
print("Probability:", probability[0])
