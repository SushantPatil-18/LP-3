# --------------------------------------------------
# BANK CUSTOMER CHURN PREDICTION USING ANN
# --------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(r"C:\Users\Vaibhav Malode\Desktop\Folder\ML\Churn_Modelling.csv")
print(df.head())
print(df.info())

# --------------------------------------------------
# 2. FEATURE & TARGET SEPARATION
# --------------------------------------------------

# Features (all except RowNumber, CustomerId, Surname, Exited)
X = df.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"])
y = df["Exited"]     # Target: 1 = Churn, 0 = Stay

# --------------------------------------------------
# 3. ENCODING CATEGORICAL COLUMNS
# --------------------------------------------------

le = LabelEncoder()
X["Gender"] = le.fit_transform(X["Gender"])          # Male/Female → 0/1
X = pd.get_dummies(X, columns=["Geography"], drop_first=True)

# --------------------------------------------------
# 4. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# 5. NORMALIZATION
# --------------------------------------------------

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------------------
# 6. BUILD ANN MODEL
# --------------------------------------------------

model = Sequential()

# Input + Hidden Layer 1
model.add(Dense(16, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.3))

# Hidden Layer 2
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3))

# Output Layer
model.add(Dense(1, activation='sigmoid'))

# Compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Show model summary
model.summary()

# --------------------------------------------------
# 7. TRAIN THE MODEL
# --------------------------------------------------

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# --------------------------------------------------
# 8. EVALUATION
# --------------------------------------------------

# Predictions
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("\nAccuracy Score:", acc)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - ANN")
plt.show()

# Classification report (Precision, Recall, F1)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
