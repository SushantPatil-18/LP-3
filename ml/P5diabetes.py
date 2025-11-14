# ------------------------------------------------------
# KNN on Diabetes Dataset
# ------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------

df = pd.read_csv("diabetes.csv")
print(df.head())
print(df.info())

# ------------------------------------------------------
# 2. Feature & Target Split
# ------------------------------------------------------

X = df.drop(columns=["Outcome"])    # all features
y = df["Outcome"]                   # target (0 or 1)

# ------------------------------------------------------
# 3. Train-Test Split
# ------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ------------------------------------------------------
# 4. Normalize Data
# ------------------------------------------------------

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------------------------------
# 5. KNN Model Training
# ------------------------------------------------------

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# ------------------------------------------------------
# 6. Predictions
# ------------------------------------------------------

y_pred = knn.predict(X_test)

# ------------------------------------------------------
# 7. Evaluation Metrics
# ------------------------------------------------------

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

# Error Rate
error_rate = 1 - accuracy
print("Error Rate:", error_rate)

# Precision
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# Recall
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# ------------------------------------------------------
# 8. Heatmap for Confusion Matrix
# ------------------------------------------------------

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - KNN Diabetes Classification")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
