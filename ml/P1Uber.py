import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv("uber.csv")
print(df.head())
print(df.info())

# -----------------------------
# 2. PRE-PROCESSING
# -----------------------------

# Remove rows with missing values
df = df.dropna()

# Convert datetime properly
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
df = df.dropna(subset=["pickup_datetime"])

# Extract features from datetime
df["year"] = df["pickup_datetime"].dt.year
df["month"] = df["pickup_datetime"].dt.month
df["day"] = df["pickup_datetime"].dt.day
df["hour"] = df["pickup_datetime"].dt.hour

# Drop unwanted columns
df = df.drop(columns=["key", "pickup_datetime", "Unnamed: 0"])

# -----------------------------
# 3. OUTLIER DETECTION
# -----------------------------
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum().sum()
print("\nOutliers detected:", outliers)

# Remove outliers using IQR
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
print("\nCleaned dataset shape:", df.shape)

# -----------------------------
# 4. CORRELATION (fixed)
# -----------------------------
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# 5. MODEL TRAINING
# -----------------------------

X = df.drop(columns=["fare_amount"])
y = df["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

# -----------------------------
# 6. MODEL EVALUATION
# -----------------------------

def evaluate_model(name, y_test, y_pred):
    print(f"\n{name} Results:")
    print("R2 Score:", r2_score(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("MAE:", mean_absolute_error(y_test, y_pred))

evaluate_model("Linear Regression", y_test, pred_lr)
evaluate_model("Random Forest", y_test, pred_rf)
