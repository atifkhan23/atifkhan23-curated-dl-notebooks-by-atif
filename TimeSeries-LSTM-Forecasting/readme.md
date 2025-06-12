# ✈️ Airline Passenger Forecasting using LSTM & Bidirectional LSTM

This project performs time series forecasting on the classic Airline Passengers dataset using deep learning models — a Stacked LSTM and a Bidirectional LSTM — built with TensorFlow/Keras. The goal is to predict monthly airline passenger numbers and compare model performances.

---

## 📊 Dataset

- **Source**: [Airline Passengers Dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv)
- **Range**: 1949 to 1960 (monthly)
- **Feature**: Monthly count of international airline passengers

---

## 🚀 Features

- Exploratory Data Analysis (EDA) with rolling statistics and seasonal decomposition
- Sequence generation and MinMax scaling
- Two LSTM architectures:
  - ✅ Stacked LSTM
  - ✅ Bidirectional LSTM
- RMSE evaluation on test set
- Forecast visualizations

---

## 📈 Model Architectures

### 1. **Stacked LSTM**
```text
LSTM (50 units, return_sequences=True)
LSTM (50 units)
Dense (1 output)
