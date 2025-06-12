# RNN Airline Passenger Prediction

This project implements a Recurrent Neural Network (RNN) using TensorFlow/Keras to forecast future airline passenger counts based on historical data. The model is trained on the classic [Airline Passengers Dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv) which contains monthly totals of international airline passengers from 1949 to 1960.

---

## 📊 Dataset

- **Source**: [Jason Brownlee's Dataset Repository](https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv)
- **Columns**:
  - `Month`: Timestamp
  - `Passengers`: Number of passengers in a given month
- **Total Samples**: 144 (Monthly data from Jan 1949 to Dec 1960)

---

## 🚀 Features

- Preprocessing with `MinMaxScaler`
- Sequence generation for time series modeling
- RNN architecture with multiple `SimpleRNN` layers and `Dropout`
- Support for optimizer switching: `Adam`, `RMSprop`, `SGD`
- Time series generation beyond the available data
- MSE evaluation and visualization of predictions

---

## 🧠 Model Architecture

```text
Input (reshaped time series)
→ SimpleRNN(50) + Dropout(0.2)
→ SimpleRNN(100) + Dropout(0.2)
→ SimpleRNN(200)
→ Dense(1)
