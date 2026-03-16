import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 1. LOAD DATA
file_path = 'pulsar_data_test.csv'
df = pd.read_csv(file_path)

# Clean column names (remove leading/trailing spaces)
df.columns = df.columns.str.strip()

print("--- Dataset Overview ---")
print(df.info())
print("\nMissing values per column:\n", df.isnull().sum())

# 2. EXPLORATORY DATA ANALYSIS (EDA)
# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.drop(columns=['target_class']).corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

# Distribution of the Mean Integrated Profile
plt.figure(figsize=(8, 5))
sns.histplot(df['Mean of the integrated profile'], kde=True, color='blue')
plt.title('Distribution of Mean Integrated Profile')
plt.show()

# 3. DATA PREPROCESSING
# Drop target_class since it is empty in the test set
X = df.drop(columns=['target_class'])

# Handle missing values (Impute with Median)
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Convert back to DataFrame for readability
X_processed = pd.DataFrame(X_scaled, columns=X.columns)

print("\n--- Preprocessing Complete ---")
print(f"Processed shape: {X_processed.shape}")
print("\nScript finished successfully. Visualizations have been generated.")