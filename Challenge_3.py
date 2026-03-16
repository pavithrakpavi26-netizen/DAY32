import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("creditcard.csv")

# Features and target
X = df.drop("Class",axis=1)
y = df["Class"]

# Split data
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# Model
model = DecisionTreeClassifier()

# Train
model.fit(X_train,y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:",accuracy_score(y_test,y_pred))