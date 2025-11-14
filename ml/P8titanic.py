# titanic_model.py
# Requirements:
# pip install pandas numpy matplotlib seaborn scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)

# -----------------------
# 1. Load data
# -----------------------
df = pd.read_csv("train.csv")
print("Raw rows:", df.shape[0])
print(df.head())

# -----------------------
# 3. Feature engineering
# -----------------------
def preprocess_df(df):
    df = df.copy()

    # Extract title from name
    df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.', expand=False).str.strip()

    # Group rare titles
    rare_titles = ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona']
    df['Title'] = df['Title'].replace(['Mlle','Ms'],'Miss')
    df['Title'] = df['Title'].replace('Mme','Mrs')
    df['Title'] = df['Title'].apply(lambda t: 'Rare' if t in rare_titles else t)

    # Cabin deck
    df['CabinDeck'] = df['Cabin'].fillna('Unknown').astype(str).str[0]
    df['CabinDeck'] = df['CabinDeck'].replace('U','Unknown')

    # Family features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Fare per person
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']

    # Drop columns not used
    df = df.drop(columns=['Name','Ticket','Cabin'])

    return df

df_proc = preprocess_df(df)
print(df_proc.head())

# -----------------------
# Split data
# -----------------------
X = df_proc.drop(columns=['Survived','PassengerId'])
y = df_proc['Survived']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train rows:", X_train.shape[0], "Validation rows:", X_val.shape[0])

# -----------------------
# Feature groups
# -----------------------
numeric_features = ['Age','Fare','FamilySize','FarePerPerson']
categorical_low_card = ['Pclass','Sex','IsAlone','Embarked']
categorical_high_card = ['Title','CabinDeck']

# -----------------------
# Pipelines
# -----------------------
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_low_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

cat_high_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, numeric_features),
    ('catlow', cat_low_pipeline, categorical_low_card),
    ('cathigh', cat_high_pipeline, categorical_high_card)
])

# -----------------------
# Logistic Regression
# -----------------------
pipe_lr = Pipeline([
    ('preproc', preprocessor),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])

pipe_lr.fit(X_train, y_train)
y_pred_lr = pipe_lr.predict(X_val)
y_proba_lr = pipe_lr.predict_proba(X_val)[:,1]

print("\nLogistic Regression Results:")
print("Accuracy:", accuracy_score(y_val, y_pred_lr))
print("Precision:", precision_score(y_val, y_pred_lr))
print("Recall:", recall_score(y_val, y_pred_lr))
print("F1:", f1_score(y_val, y_pred_lr))
print("ROC AUC:", roc_auc_score(y_val, y_proba_lr))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred_lr))

# -----------------------
# Random Forest
# -----------------------
pipe_rf = Pipeline([
    ('preproc', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=200, random_state=42))
])

pipe_rf.fit(X_train, y_train)
y_pred_rf = pipe_rf.predict(X_val)
y_proba_rf = pipe_rf.predict_proba(X_val)[:,1]

print("\nRandom Forest Results:")
print("Accuracy:", accuracy_score(y_val, y_pred_rf))
print("Precision:", precision_score(y_val, y_pred_rf))
print("Recall:", recall_score(y_val, y_pred_rf))
print("F1:", f1_score(y_val, y_pred_rf))
print("ROC AUC:", roc_auc_score(y_val, y_proba_rf))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred_rf))

# -----------------------
# Save model
# -----------------------
with open("titanic_rf_model.pkl", "wb") as f:
    pickle.dump(pipe_rf, f)

print("\nModel saved: titanic_rf_model.pkl")
