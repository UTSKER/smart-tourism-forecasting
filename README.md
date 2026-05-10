# Tourism Forecasting for Tirupati Balaji Temple (TTD, Andhra Pradesh)

> **Applied Forecasting Methods (IT-402) — Final Project**
> **Dhirubhai Ambani University** *(Formerly DA-IICT)*

---

## Course Information

| Field              | Details                          |
| ------------------ | -------------------------------- |
| **Course**         | Applied Forecasting Methods      |
| **Course Code**    | IT-402                           |
| **Instructor**     | Pritam Anand                     |
| **Project Title**  | Tourism Forecasting for Tirupati Balaji Temple |
| **Group Number**   | 14                               |
| **Submission Date**| May 5, 2026                      |

---

## Group Members

| Name               | Student ID  |
| ------------------ | ----------- |
| Utsker Dhameliya   | 202301083   |
| Smit Limbasiya     | 202301139   |

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models Used](#models-used)
- [Results Summary](#results-summary)
- [Forecast-to-Action Framework](#forecast-to-action-framework)
- [Dependencies](#dependencies)
- [How to Run](#how-to-run)
- [Future Work](#future-work)
- [References](#references)

---

## Project Overview

This project presents a complete end-to-end time-series forecasting pipeline for predicting the **daily number of pilgrims (darshans) visiting the Tirupati Balaji Temple** (Tirumala Tirupati Devasthanam — TTD), one day in advance.

The Tirumala–Tirupati pilgrimage system is one of the most crowded and continuously operating religious service environments in India. Every day, thousands of devotees visit the temple, and the numbers shift due to festivals, seasons, weekends, and special events. Accurate day-ahead forecasting is critical for:

- **Crowd control and queue management**
- **Prasadam and laddu production planning**
- **Transport and bus deployment (APSRTC)**
- **Accommodation and staffing allocation**
- **Security and sanitation preparedness**

The project compares classical statistical models (AR, ARIMA, SARIMAX) with modern deep learning approaches (RNN, LSTM, GRU, TCN) and probabilistic quantile regression — all evaluated under a strict chronological train/validation/test split to prevent data leakage.

---

## Problem Statement

The Tirupati Balaji Temple receives a highly variable number of visitors daily. This variability is driven by:

- Weekends and public holidays
- Religious festivals and special events
- Weather and seasonal conditions
- Google Trends / public interest signals

The goal is to build a accurate, data-driven forecasting system that minimizes both types of errors while **prioritizing safety** by penalizing underprediction more heavily.

---

## Dataset

| Property             | Value                                                                 |
| -------------------- | --------------------------------------------------------------------- |
| **Source**           | Kaggle — Tirumala Tirupati Devasthanam Darshan Dataset                |
| **Dataset Link**     | https://www.kaggle.com/datasets/vishnumadhav2454/tirumala-tirupati-devasthanam-darshan |
| **Format**           | Daily CSV — one row per day                                           |
| **Date Range**       | 16 October 2013 to 06 July 2025                                      |
| **Total Records**    | 3,572 days (raw); 1,222 records after preprocessing                   |
| **Total Columns**    | 17 (raw)                                                              |
| **Target Variable**  | Daily darshan (pilgrim) count                                         |
| **Cleaned Output**   | `ttd_darshans.csv`                                                    |

### Key Features

| Feature                 | Description                                              |
| ----------------------- | -------------------------------------------------------- |
| `darshans`              | Daily pilgrim count (target variable)                    |
| `Weekend`               | Binary flag — 1 if Saturday or Sunday                    |
| `Holiday`               | Binary flag — 1 if public/religious holiday              |
| `darshans_lag1`         | Darshan count from the previous day                      |
| `darshans_lag7`         | Darshan count from 7 days ago                            |
| `rolling_avg_7_safe`    | 7-day rolling average using shifted (leakage-free) data  |
| `temp_max_lag1`         | Previous day's maximum temperature                       |
| `temp_range_lag1`       | Previous day's temperature range                         |
| `rainfall_lag1 (log)`   | Log-transformed previous day's rainfall                  |
| `google_trends_lag`     | Lagged Google Trends interest score                      |

### Descriptive Statistics (Preprocessed Dataset)

| Statistic  | Mean       | Std Dev  | Min    | Max    | Median    |
| ---------- | ---------- | -------- | ------ | ------ | --------- |
| Darshans   | 70,413     | 8,868    | 40,265 | 95,080 | 70,082    |

---

## Methodology

### Data Preprocessing

- Converted date column to datetime format and sorted chronologically
- Removed leakage-prone features (precomputed rolling averages, current-day weather)
- Created lag features: `darshans_lag1`, `darshans_lag7`, `rolling_avg_7_safe`
- Applied `log1p()` transformation to rainfall data to reduce skewness
- Removed initial rows with undefined lag values; remaining missing values filled with forward-fill
- Final clean dataset stored as `ttd_darshans.csv`

### Train–Validation–Test Split

A strict **chronological split** is applied to prevent data leakage:

```
Test Set:        Last 30 days   → Final evaluation
Validation Set:  60 days before test → Hyperparameter tuning
Training Set:    All remaining earlier data → Model fitting
```

### Forecasting Strategy

- **One-step-ahead rolling forecast**: the model predicts `t+1` using all data up to `t`
- Once the actual value is observed, it is incorporated into the dataset before the next prediction
- This mimics real-world deployment and prevents error accumulation

### Stationarity Analysis

- ADF and KPSS tests were applied to evaluate stationarity
- Regular differencing was **not** applied (`d = 0`) due to deterministic (not stochastic) trends
- Seasonal differencing (`D = 1, s = 7`) was retained to capture weekly patterns

### Exploratory Data Analysis

Key findings from EDA:

- **Average daily darshans**: ~70,413 visitors (range: 40,265 – 95,080)
- **Weekly seasonality**: Saturdays and Sundays peak at ~77,630 and ~79,237 respectively; Thursdays are lowest at ~62,605
- **Yearly trend**: Stable across 2022–2025 (range: 69,718 – 71,549), confirming post-COVID stabilization
- **ACF/PACF**: Strong spikes at lags 1 and 7 confirm short-term autocorrelation and weekly seasonality
- **Seasonal decomposition**: Clear 7-day cycle; residual component captures festival/event-driven spikes

---

## Models Used

### Statistical Models

| Model       | Description                                             |
| ----------- | ------------------------------------------------------- |
| **AR**      | Autoregressive — uses past observations                 | 
| **MA**      | Moving Average — uses past forecast errors              | 
| **ARMA**    | Combination of AR and MA                                | 
| **ARIMAX**  | ARMA with exogenous variables                           | 
| **SARIMAX** | ARIMAX with explicit weekly seasonal components (s=7)   |      

### Deep Learning Models

| Model              | Architecture                          | Window Size |
| ------------------ | ------------------------------------- | ----------- |
| **LSTM**           | 2 layers (64, 32 units)               | 21 days     |
| **Simple RNN**     | 1 layer (64 units)                    | 21 days     |
| **GRU + Attention**| 2 layers (128, 64 units)              | 21 days     |
| **TCN**            | Kernel=3, Dilations=[1,2,4,8,16]      | —           |
| **LightGBM**       | Gradient boosting with lag features   | —           |

### Probabilistic Model

- **Quantile Regression** with asymmetric pinball loss
- Produces Q05, Q50 (median), and Q95 forecasts with an 80% prediction interval
- Designed to penalize underprediction more heavily than overprediction

---

## Results Summary

### Statistical Models

| Model           | Config       | Val MAE  | Val RMSE | Val MAPE | Test MAE | Test RMSE | Test MAPE |
| --------------- | ------------ | -------- | -------- | -------- | -------- | --------- | --------- |
| AR              | AR(7)        | 4688.70  | 5496.77  | 6.35%    | 3198.84  | 4022.17   | 3.92%     |
| MA              | MA(6)        | 5961.02  | 7543.40  | 8.01%    | 5986.83  | 8011.79   | 7.15%     |
| ARMA            | ARMA(7,6)    | 3783.02  | 4772.40  | 5.12%    | 2455.83  | 3163.81   | 2.98%     |
| ARIMAX          | (7,0,6)      | 3648.63  | 4525.25  | 5.03%    | 2834.69  | 3484.10   | 3.49%     |
| **SARIMAX**  | (6,0,6)(1,1,1,7) | **3552.09** | **4538.21** | **4.88%** | **1921.92** | **2557.89** | **2.38%** |

### Deep Learning Models

| Model           | Val MAE  | Val MAPE | Test MAE | Test MAPE |
| --------------- | -------- | -------- | -------- | --------- |
| LSTM            | 4259.99  | 5.86%    | 4422.46  | 5.50%     |
| LightGBM        | 4819.94  | 6.72%    | 3983.89  | 5.04%     |
| Simple RNN      | 4689.44  | 6.50%    | 4227.97  | 5.23%     |
| TCN             | 3968.74  | 5.47%    | 3917.45  | 4.87%     |
| GRU + Attention | 4231.87  | 5.94%    | 3947.79  | 4.91%     |

### Quantile Regression — Findings & Interpretation

The quantile regression model was trained using an **asymmetric pinball loss**, intentionally
penalizing underprediction more heavily than overprediction — reflecting the real-world reality
that resource shortages at Tirupati carry far greater risk (overcrowding, stampede, health hazards)
than surplus. The model forecasts three quantiles (Q05, Q50, Q95), forming a **90% prediction band**
with an average width of ~8,000 devotees. While this interval may appear wide, it is narrower than
one standard deviation of the data itself (σ = 8,868), confirming the model **does add predictive
value** over a naive baseline. In practical terms: *on any given day, we are 90% confident the
actual pilgrim count will fall within ~8,000 devotees of the median forecast — shifting between
~66,000–74,000 on typical weekdays and ~81,000–89,000 on weekend/festival peaks.* The model's
directional learning is evident — underpredictions dropped sharply from **18 (validation) to just 2
(test)**, while coverage improved from 33.3% to 66.7% against an 80% target. The remaining coverage
gap is not a calibration failure — Ljung-Box testing (p < 0.05) confirmed that residuals still carry
unexplained structure, primarily driven by sudden festival and event-day spikes that no lag, weather,
or calendar feature in the current dataset can anticipate. This is the core **research gap**: the
interval inflates globally to compensate for these unpredictable high-variance days. Incorporating
event-intensity features and festival schedules in future work is expected to tighten the band
significantly on normal days while improving coverage on spike days.

### Key Takeaways

- **SARIMAX** is the best-performing model overall with a test MAPE of **2.38%**
- Deep learning models underperform classical methods due to the limited dataset size (~1,222 records)
- The **7-day weekly seasonality** is the strongest signal across all models
- **Quantile regression** successfully shifts predictions toward safer overestimates, reducing underpredictions from 18 (validation) to just 2 (test)

---

## Forecast-to-Action Framework

The model is not just a prediction engine — it serves as a **decision-support system** for all stakeholders:

| Stakeholder      | Action on High Forecast            | Action on Low Forecast               |
| ---------------- | ---------------------------------- | ------------------------------------ |
| TTD              | Increase prasadam & laddu production, add staff | Reduce batch cooking, flexible shifts |
| APSRTC           | Deploy additional buses in phases  | Reduce bus frequency, save fuel      |
| Police/Security  | Strengthen crowd management        | Standard deployment                  |
| Hotels/Vendors   | Increase inventory, dynamic pricing | Reduce stock, manage vacancy         |

### Economic Impact of Forecast Errors

| Error Type        | Cost Per Devotee | Includes                                          |
| ----------------- | ---------------- | ------------------------------------------------- |
| Under-forecast    | Rs. 298–308      | Food, laddu, overtime staff, transport, hotels    |
| Over-forecast     | Rs. 275–283      | Wasted food, idle transport, vacant rooms         |

### Threshold Triggers

| Forecast Range | Alert | Action |
|----------------|-------|--------|
| < 55,000 | 🟢 Low | Minimal staffing, reduced buses |
| 55,000–75,000 | 🟡 Normal | Standard operations |
| 75,000–85,000 | 🟠 High | Extra buses, additional prasadam batch |
| > 85,000 | 🔴 Critical | Full emergency protocol, max deployment |

> **Safety Note**: Under-forecasting is treated as more critical due to overcrowding, potential stampede risks, and health hazards for pilgrims. The forecasting strategy is intentionally biased toward slight overestimation.

---

## Dependencies

Install all required packages:

```bash
pip install numpy pandas matplotlib seaborn statsmodels scikit-learn lightgbm tensorflow
```

| Library        | Purpose                                         |
| -------------- | ----------------------------------------------- |
| `numpy`        | Numerical operations                            |
| `pandas`       | Data manipulation and time-series handling      |
| `matplotlib`   | Plotting and visualization                      |
| `seaborn`      | Statistical visualization                       |
| `statsmodels`  | AR, ARMA, ARIMAX, SARIMAX, QuantReg models      |
| `scikit-learn` | Preprocessing and evaluation utilities          |
| `lightgbm`     | Gradient boosting model                         |
| `tensorflow`   | LSTM, GRU, RNN, TCN deep learning models        |

---

## How to Run

### Google Colab (Recommended)

The full implementation is available and executable on Google Colab:

🔗 [Open in Google Colab](https://colab.research.google.com/drive/15sZkjXry9P3H0qLcykpAGFY1odtxLa9O?usp=sharing)

The notebook includes all steps: data preprocessing, EDA, model training, evaluation, and probabilistic forecasting.

### Running Locally

```bash
# Clone the repository
git clone <repo-url>
cd <repo-folder>

# Install dependencies
pip install numpy pandas matplotlib seaborn statsmodels scikit-learn lightgbm tensorflow

# Download the dataset from Kaggle and place it in the project root
# https://www.kaggle.com/datasets/vishnumadhav2454/tirumala-tirupati-devasthanam-darshan

# Run the notebook
jupyter notebook forecasting.ipynb
```

### Expected Outputs

| Output                    | Description                                              |
| ------------------------- | -------------------------------------------------------- |
| `ttd_darshans.csv`        | Cleaned, leakage-free dataset ready for modeling         |
| Inline EDA plots          | 15+ visualizations of temporal patterns and statistics   |
| Model comparison table    | Validation and test metrics for all models               |
| Quantile forecast plot    | 80% prediction interval band with Q05, Q50, Q95 curves  |

---

## Future Work

- Introduce controlled upward bias in SARIMAX (e.g., `ŷ_biased = ŷ + c`) for risk-aware forecasting
- Integrate real-time weather APIs and live festival calendars as dynamic exogenous inputs
- Develop a full-stack deployment (frontend + backend) as a real-time decision-support platform
- Implement role-based dashboards for TTD, transport authorities, hotels, and vendors
- Extend outputs to actionable variables: laddu count, prasadam batches, bus count, room allocation
- Incorporate event-intensity features and crowd-trigger indicators to capture unexplained spikes
- Explore hybrid models combining SARIMAX with residual correction via deep learning

---

## References

1. R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed. OTexts, 2021. https://otexts.com/fpp3/
2. G. E. P. Box, G. M. Jenkins, and G. C. Reinsel, *Time Series Analysis: Forecasting and Control*, 5th ed. Wiley, 2015.
3. S. Makridakis et al., "The M5 Competition," *International Journal of Forecasting*, vol. 38, no. 4, 2022.
4. Tirumala Tirupati Devasthanams — https://www.tirumala.org
5. Kaggle Dataset — https://www.kaggle.com/datasets/vishnumadhav2454/tirumala-tirupati-devasthanam-darshan

---

<div align="center">

**Group 14 — Applied Forecasting Methods (IT-402)**
**Dhirubhai Ambani University (Formerly DA-IICT)**

Utsker Dhameliya · Smit Limbasiya

</div>
