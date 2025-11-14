# STOCK ANALYSIS & RETURNS PREDICTION (Indian market 2000-2020)
# Requirements:
# pip install pandas numpy matplotlib seaborn ta pmdarima statsmodels scikit-learn tensorflow

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
plt.style.use("seaborn-v0_8-darkgrid")

# ML & Stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm

# LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Technical indicators helper
import ta

# ----------------------------
# User-configurable
# ----------------------------
DATAFILE = "ADANIPOWER.NS.csv"   # <- change if needed
DATE_COL = "Date"
PRICE_COL_CANDIDATES = ["Close","Adj Close","Close Price","close"]

# ----------------------------
# 1) Load dataset
# ----------------------------
df = pd.read_csv(DATAFILE)
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.sort_values(DATE_COL).reset_index(drop=True)

# pick price column
price_col = None
for c in PRICE_COL_CANDIDATES:
    if c in df.columns:
        price_col = c
        break
if price_col is None:
    # fallback: choose numeric column besides Date
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in []]
    if len(numeric_cols) == 0:
        raise ValueError("No numeric price-like column found. Please provide 'Close' or similar.")
    price_col = numeric_cols[0]
print("Using price column:", price_col)

df = df[[DATE_COL, price_col] + [c for c in df.columns if c not in [DATE_COL, price_col]]]
df.rename(columns={price_col: "Close"}, inplace=True)

print("Data range:", df[DATE_COL].min(), "to", df[DATE_COL].max())
print("Rows:", len(df))

# ----------------------------
# 2) Clean & Basic EDA
# ----------------------------
# check missing
print("\nMissing values:\n", df.isna().sum())

# fill/forward-fill Close
df["Close"] = df["Close"].ffill().bfill()
df = df.dropna(subset=["Close"]).reset_index(drop=True)

# compute returns
df["Return"] = df["Close"].pct_change()
df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
df = df.dropna().reset_index(drop=True)

# ups and downs summary
df["Direction"] = (df["Return"] > 0).astype(int)
ups = int(df["Direction"].sum())
downs = int((df.shape[0] - ups))
print(f"Up days: {ups}, Down days: {downs}, Up ratio: {ups/len(df):.3f}")

# moving averages & volatility
df["MA20"] = df["Close"].rolling(20).mean()
df["MA50"] = df["Close"].rolling(50).mean()
df["Vol30"] = df["LogReturn"].rolling(30).std() * np.sqrt(252)

# drawdowns
df["CumRet"] = (1 + df["Return"]).cumprod()
df["RollMax"] = df["CumRet"].cummax()
df["Drawdown"] = df["CumRet"] / df["RollMax"] - 1

# plots
plt.figure(figsize=(14,6))
plt.plot(df[DATE_COL], df["Close"], label="Close")
plt.plot(df[DATE_COL], df["MA20"], label="MA20")
plt.plot(df[DATE_COL], df["MA50"], label="MA50")
plt.title("Price and Moving Averages")
plt.legend(); plt.show()

plt.figure(figsize=(12,4))
sns.histplot(df["LogReturn"], bins=150, kde=True)
plt.title("Log returns distribution"); plt.show()

plt.figure(figsize=(12,4))
plt.plot(df[DATE_COL], df["Drawdown"])
plt.title("Drawdown"); plt.show()

# ----------------------------
# 3) Feature engineering (technical indicators)
# ----------------------------
df["rsi14"] = ta.momentum.rsi(df["Close"], window=14)
macd = ta.trend.MACD(df["Close"])
df["macd"] = macd.macd()
df["macd_signal"] = macd.macd_signal()
bb = ta.volatility.BollingerBands(df["Close"], window=20)
df["bb_high"] = bb.bollinger_hband()
df["bb_low"]  = bb.bollinger_lband()
df["bb_pct"]  = (df["Close"] - df["bb_low"]) / (df["bb_high"] - df["bb_low"] + 1e-9)

# lagged returns
for lag in [1,2,3,5,10]:
    df[f"r_lag_{lag}"] = df["LogReturn"].shift(lag)

# drop NaNs produced by indicators
df = df.dropna().reset_index(drop=True)
print("After indicators rows:", len(df))

# ----------------------------
# 4) SARIMAX on returns (statistical forecasting)
# ----------------------------
# target: LogReturn
target = "LogReturn"
train_size = int(0.8 * len(df))
train = df.iloc[:train_size].copy()
test  = df.iloc[train_size:].copy()

print("SARIMAX training rows:", len(train), "test rows:", len(test))

# auto_arima selects order
arima_auto = pm.auto_arima(train[target], seasonal=False, stepwise=True, suppress_warnings=True,
                           error_action="ignore", max_p=5, max_q=5)
print("Selected ARIMA order:", arima_auto.order)

sarimax = SARIMAX(train[target], order=arima_auto.order, enforce_stationarity=False, enforce_invertibility=False)
res = sarimax.fit(disp=False)
print(res.summary())

# forecast returns for test period
sarimax_fore = res.get_forecast(steps=len(test))
pred_returns = sarimax_fore.predicted_mean
pred_returns.index = test.index

# evaluate SARIMAX on returns
def rmse(a,b): return np.sqrt(mean_squared_error(a,b))
sar_rmse = rmse(test[target], pred_returns)
sar_mae  = mean_absolute_error(test[target], pred_returns)
print("SARIMAX RMSE (returns):", sar_rmse, "MAE:", sar_mae)

# directional accuracy (returns >0)
pred_dir = (pred_returns > 0).astype(int)
dir_acc = (pred_dir.values == test["Direction"].values).mean()
print("SARIMAX directional accuracy:", dir_acc)

# plot returns actual vs pred
plt.figure(figsize=(12,4))
plt.plot(train[DATE_COL], train[target], label="train")
plt.plot(test[DATE_COL], test[target], label="actual")
plt.plot(test[DATE_COL], pred_returns, label="sarimax_pred")
plt.legend(); plt.title("SARIMAX - LogReturn Forecast"); plt.show()

# ----------------------------
# 5) LSTM on price (sequence model)
# ----------------------------
# We'll predict next-day Close using past window of Close
series = df["Close"].values.reshape(-1,1)
scaler = MinMaxScaler()
series_s = scaler.fit_transform(series)

SEQ_LEN = 60
def create_seq(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_all, y_all = create_seq(series_s, SEQ_LEN)
dates_seq = df[DATE_COL].iloc[SEQ_LEN:].reset_index(drop=True)

# compute sequence split index by date
split_date = df.iloc[train_size][DATE_COL]
seq_split = np.where(dates_seq >= split_date)[0][0]
X_train_seq, X_test_seq = X_all[:seq_split], X_all[seq_split:]
y_train_seq, y_test_seq = y_all[:seq_split], y_all[seq_split:]

# reshape for LSTM
X_train_seq = X_train_seq.reshape((X_train_seq.shape[0], SEQ_LEN, 1))
X_test_seq  = X_test_seq.reshape((X_test_seq.shape[0], SEQ_LEN, 1))

print("LSTM shapes:", X_train_seq.shape, X_test_seq.shape)

tf.keras.backend.clear_session()
model = Sequential([
    LSTM(64, input_shape=(SEQ_LEN,1), return_sequences=False),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")
history = model.fit(X_train_seq, y_train_seq, epochs=25, batch_size=32, validation_split=0.1, verbose=1)

# predictions
y_pred_s = model.predict(X_test_seq).flatten()
y_test_inv = scaler.inverse_transform(y_test_seq.reshape(-1,1)).flatten()
y_pred_inv = scaler.inverse_transform(y_pred_s.reshape(-1,1)).flatten()

# evaluate price predictions
print("LSTM price RMSE:", rmse(y_test_inv, y_pred_inv), "MAE:", mean_absolute_error(y_test_inv, y_pred_inv))

# directional accuracy (by comparing next-day price change)
actual_dir = (np.diff(y_test_inv) > 0).astype(int)
pred_dir_lstm = (np.diff(y_pred_inv) > 0).astype(int)
dir_acc_lstm = (actual_dir == pred_dir_lstm).mean()
print("LSTM directional accuracy (approx):", dir_acc_lstm)

plt.figure(figsize=(12,4))
plt.plot(dates_seq.iloc[seq_split:].values, y_test_inv, label="Actual Close")
plt.plot(dates_seq.iloc[seq_split:].values, y_pred_inv, label="LSTM Pred")
plt.legend(); plt.title("LSTM price predictions"); plt.show()

# ----------------------------
# 6) RandomForest classifier for next-day direction (features -> up/down)
# ----------------------------
# Use technical features + lag returns to predict next-day direction
features = ["rsi14","macd","macd_signal","bb_pct","MA20","MA50","Vol30","r_lag_1","r_lag_2","r_lag_3"]
# ensure features exist
features = [f for f in features if f in df.columns]
X = df[features]
y = df["Direction"].shift(-1)  # next-day direction
X = X[:-1]; y = y[:-1]   # drop last NaN target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# scale
scaler_rf = StandardScaler()
X_train_s = scaler_rf.fit_transform(X_train)
X_test_s  = scaler_rf.transform(X_test)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_s, y_train)
y_pred_rf = rf.predict(X_test_s)

print("RF accuracy:", accuracy_score(y_test, y_pred_rf))
print("RF confusion matrix:\n", confusion_matrix(y_test, y_pred_rf))

# ----------------------------
# 7) Save outputs
# ----------------------------
out = pd.DataFrame({
    "Date": test[DATE_COL],
    "Actual_LogReturn": test["LogReturn"].values,
    "SARIMAX_Pred_LogReturn": pred_returns.values
})
out.to_csv("sarimax_returns_pred.csv", index=False)

lstm_out = pd.DataFrame({
    "Date": dates_seq.iloc[seq_split:].values,
    "Actual_Close": y_test_inv,
    "LSTM_Pred_Close": y_pred_inv
})
lstm_out.to_csv("lstm_price_pred.csv", index=False)

print("Saved predictions to 'sarimax_returns_pred.csv' and 'lstm_price_pred.csv'")

# ----------------------------
# 8) Summary of results
# ----------------------------
print("\nSUMMARY:")
print(f"- SARIMAX (returns): RMSE={sar_rmse:.6f}, MAE={sar_mae:.6f}, DirectionalAcc={dir_acc:.3f}")
print(f"- LSTM (price): RMSE={rmse(y_test_inv, y_pred_inv):.3f}, DirectionalAcc≈{dir_acc_lstm:.3f}")
print(f"- RF direction classifier accuracy: {accuracy_score(y_test, y_pred_rf):.3f}")
print("Files saved for closer inspection.")

# End of script
