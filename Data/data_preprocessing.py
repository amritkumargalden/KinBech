import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the CSV file
df = pd.read_csv('hamrobazaar_extracted_products.csv')

print(df.head())  # Display the first few rows of the DataFrame
print(df.info())  # Display information about the DataFrame, including data types and non-null counts
print(df.describe())  # Display summary statistics for numerical columns
print(df.isnull().sum())  # Display the count of missing values for each column

# data_preprocessing.py (continued)

# Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Remove outliers (asking_price should be reasonable)
Q1 = df['asking_price'].quantile(0.25)
Q3 = df['asking_price'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['asking_price'] >= Q1 - 1.5*IQR) & (df['asking_price'] <= Q3 + 1.5*IQR)]

# Feature engineering
df['depreciation_rate'] = (df['original_price'] - df['asking_price']) / df['original_price']
df['age_score'] = df['used_months'] / 12  # Convert to years
df['value_retention'] = df['asking_price'] / df['original_price']

# Select features for training
X = df[['original_price', 'used_months', 'condition_score', 'depreciation_rate', 'value_retention']]
y = df['asking_price']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Dataset shape: {X_scaled.shape}")
print(f"Target variable (asking_price) statistics:\n{y.describe()}")

# Save scaler for later use
import pickle
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))
