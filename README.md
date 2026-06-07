# Microsoft Stock Price Prediction – LSTM Neural Network

A deep learning model that predicts Microsoft stock closing prices using a stacked LSTM (Long Short-Term Memory) neural network trained on 5 years of historical data. Built with TensorFlow/Keras, the model learns temporal dependencies in price sequences to forecast future closing prices.

---

## Results

| Metric | Value |
|---|---|
| Model | Stacked LSTM (2 layers, 64 units each) |
| Dataset | Microsoft Stock (2013–2018) |
| Train/Test Split | 95% / 5% |
| Sequence Length | 60-day rolling window |
| Loss Function | Mean Squared Error (MSE) |
| Optimizer | Adam |
| Epochs | 20 |
| Batch Size | 32 |

---

## Model Architecture

```
Input: (60 timesteps, 1 feature)
        ↓
LSTM(64, return_sequences=True)
        ↓
LSTM(64)
        ↓
Dense(128)
        ↓
Dropout(0.5)
        ↓
Dense(1)  →  Predicted closing price
```

- **Stacked LSTMs** — two LSTM layers allow the model to learn both low-level temporal patterns and higher-level sequential dependencies in price data
- **60-day sliding window** — each prediction is conditioned on the previous 60 trading days, capturing medium-term price trends
- **Dropout (0.5)** — reduces overfitting by randomly deactivating 50% of neurons during training
- **StandardScaler normalization** — stabilizes training by scaling price values to zero mean and unit variance before feeding into the network

---

## Techniques Used

- **Time Series Forecasting** — framed as a supervised learning problem using sliding window sequences
- **Feature Normalization** — StandardScaler applied to closing prices before training; inverse transform applied to predictions for interpretability
- **Sequence Construction** — rolling 60-day windows built for both train and test sets to preserve temporal order
- **Exploratory Data Analysis (EDA)** — visualized open/close price trends, trading volume over time, and feature correlation heatmap before modeling
- **Train/Test Split** — strict temporal split (no shuffling) to prevent data leakage from future prices into training

---

## Visualizations

The script generates 4 plots:

1. **Open vs Close Price over time** — trend analysis across the full dataset
2. **Trading Volume over time** — volume pattern analysis
3. **Feature Correlation Heatmap** — pairwise correlation across all numeric features (open, close, high, low, volume)
4. **Predicted vs Actual Closing Price** — overlays train, actual test, and predicted test prices on a single chart

---

## Project Structure

```
stock-price-lstm/
│
├── stock_prediction.py     # Full pipeline: EDA, preprocessing, model training, evaluation
├── requirements.txt        # Dependencies
└── README.md
```

---

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Requirements

```
tensorflow
pandas
numpy
matplotlib
seaborn
scikit-learn
```

### Dataset

Download the Microsoft Stock dataset (e.g. from [Kaggle](https://www.kaggle.com/)) and update the file path in the script:

```python
microsoft = pd.read_csv('path/to/MicrosoftStock.csv')
```

### Run

```bash
python stock_prediction.py
```

---

## Key Design Decisions

**Why LSTM over standard RNN?**
Standard RNNs suffer from vanishing gradients over long sequences, making them unable to learn dependencies beyond a few timesteps. LSTMs use gated memory cells (input, forget, output gates) to retain relevant information over the full 60-day window — critical for capturing multi-week price trends.

**Why a 60-day window?**
60 trading days (~3 months) is a standard lookback period in quantitative finance, balancing short-term noise with enough historical context for the model to detect meaningful trends.

**Why StandardScaler over MinMaxScaler?**
StandardScaler is more robust to outliers in financial data — extreme price spikes won't compress the rest of the data into a tiny range the way MinMaxScaler would.

**Why Dropout(0.5)?**
Stock price data is inherently noisy. A 50% dropout rate aggressively regularizes the Dense layer, preventing the model from memorizing training patterns that don't generalize.

---

## Skills Demonstrated

`Python` `TensorFlow` `Keras` `LSTM` `Deep Learning` `Time Series Forecasting` `Neural Networks` `Feature Engineering` `Data Preprocessing` `StandardScaler` `Dropout Regularization` `Exploratory Data Analysis` `Pandas` `NumPy` `Matplotlib` `Seaborn` `scikit-learn`

---

## Author

**Tamanna Subudhi**
B.S. Computer Science (AI Concentration) — Purdue University Northwest
[LinkedIn](https://linkedin.com/in/tamanna-subudhi-6792a026a)
