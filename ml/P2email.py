# -------------------------------------------
# EMAIL SPAM CLASSIFICATION USING KNN & SVM
# -------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------
# 1. LOAD DATA
# --------------------------
df = pd.read_csv(r"C:\Users\Vaibhav Malode\Desktop\Folder\ML\emails.csv")
print(df.head())
print(df.info())

# --------------------------
# 2. PREPROCESSING
# --------------------------

# Remove column 'Email No.' if present
if "Email No." in df.columns:
    df = df.drop(columns=["Email No."])

# Separate features and target
X = df.drop(columns=["Prediction"])
y = df["Prediction"]    # 1 = Spam, 0 = Not spam

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------
# 3. KNN MODEL
# --------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
pred_knn = knn.predict(X_test)

# --------------------------
# 4. SVM MODEL
# --------------------------
svm = SVC(kernel='linear', C=1)
svm.fit(X_train, y_train)
pred_svm = svm.predict(X_test)

# --------------------------
# 5. EVALUATION FUNCTION
# --------------------------
def evaluate_model(name, y_test, y_pred):
    print(f"\n===== {name} RESULTS =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# --------------------------
# 6. SHOW RESULTS
# --------------------------
evaluate_model("K-Nearest Neighbors", y_test, pred_knn)
evaluate_model("Support Vector Machine", y_test, pred_svm)
