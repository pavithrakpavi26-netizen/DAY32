import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. LOAD DATA
try:
    df = pd.read_csv('test.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'test.csv' not found. Ensure it's in the same folder as this script.")
    exit()

# 2. DATA INSPECTION
print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Data Information ---")
print(df.info())

# 3. EXPLORATORY DATA ANALYSIS (EDA)
# Let's visualize the distribution of RAM and Battery Power
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df['ram'], kde=True, color='blue')
plt.title('Distribution of RAM')

plt.subplot(1, 2, 2)
sns.histplot(df['battery_power'], kde=True, color='green')
plt.title('Distribution of Battery Power')

plt.tight_layout()
plt.show()

# Correlation Heatmap (To see which features are related)
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# 4. DATA PREPROCESSING
# Usually, we drop 'id' as it doesn't help with predictions
if 'id' in df.columns:
    X = df.drop(columns=['id'])
else:
    X = df

# Feature Scaling (Important for models like SVM or KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n--- Preprocessing Complete ---")
print(f"Features scaled. Shape: {X_scaled.shape}")