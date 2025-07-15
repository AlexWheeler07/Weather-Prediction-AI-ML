# Weather Prediction Machine Learning Project

## Objective

Build a regression model to predict temperature using weather-related features such as humidity, pressure, wind speed, visibility, and apparent temperature.

---

## Dataset Used

https://www.kaggle.com/datasets/muthuj7/weather-dataset

---

## Project Structure

```
weather_prediction_project/
├── data/
│   └── my_weather_data.csv      # CSV file containing the dataset
├── models/
│   └── model.pkl                # (optional) saved model
├── weather_predictor.py         # main script for training, evaluation, and visualization
├── requirements.txt             # Python dependencies
└── README.md
```

---

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Add Dataset

Place your weather dataset CSV as `my_weather_data.csv` inside the `data/` folder. The expected columns are:

- `humidity`
- `pressure`
- `wind_speed`
- `visibility`
- `apparent_temperature`
- `temperature` _(target)_

> Make sure the dataset has no missing values in the above columns. The script also removes outliers using IQR.

### 3. Run the Script

```bash
python weather_predictor.py
```

---

## Features Used

| Feature                | Description                 |
| ---------------------- | --------------------------- |
| `humidity`             | Relative humidity (%)       |
| `pressure`             | Atmospheric pressure (hPa)  |
| `wind_speed`           | Wind speed (km/h)           |
| `visibility`           | Visibility distance (km)    |
| `apparent_temperature` | Feels-like temperature (°C) |

---

## Machine Learning Model

- **Algorithm**: Random Forest Regressor (100 trees)
- **Target Variable**: `temperature`
- **Evaluation Metrics**:

  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - R² Score (Coefficient of Determination)

---

## Key Capabilities

- Data loading and cleaning
- Outlier removal using IQR
- Exploratory data analysis (EDA)
- Visualizations (correlation matrix, scatter plots)
- Train-test split (80/20)
- Model training and evaluation
- Visualization of predicted vs actual temperatures

---

## Output

```
Loading data from data/my_weather_data.csv

Exploratory Data Analysis
==================================================
Shape: (96453, 12)
Columns: ['formatted_date', 'summary', 'precip_type', 'temperature', 'apparent_temperature', 'humidity', 'wind_speed', 'wind_bearing_degrees', 'visibility', 'loud_cover', 'pressure', 'daily_summary']

Summary Statistics:
       temperature  apparent_temperature  humidity  wind_speed  wind_bearing_degrees  visibility  loud_cover  pressure
count     96453.00              96453.00  96453.00    96453.00              96453.00    96453.00     96453.0  96453.00
mean         11.93                 10.86      0.73       10.81                187.51       10.35         0.0   1003.24
std           9.55                 10.70      0.20        6.91                107.38        4.19         0.0    116.97
min         -21.82                -27.72      0.00        0.00                  0.00        0.00         0.0      0.00
25%           4.69                  2.31      0.60        5.83                116.00        8.34         0.0   1011.90
50%          12.00                 12.00      0.78        9.97                180.00       10.05         0.0   1016.45
75%          18.84                 18.84      0.89       14.14                290.00       14.81         0.0   1021.09
max          39.91                 39.34      1.00       63.85                359.00       16.10         0.0   1046.38

Missing values:
formatted_date            0
summary                   0
precip_type             517
temperature               0
apparent_temperature      0
humidity                  0
wind_speed                0
wind_bearing_degrees      0
visibility                0
loud_cover                0
pressure                  0
daily_summary             0
dtype: int64

Feature Correlations with Temperature:
temperature             1.000
apparent_temperature    0.993
visibility              0.393
wind_bearing_degrees    0.030
wind_speed              0.009
pressure               -0.005
humidity               -0.632
loud_cover                NaN
Name: temperature, dtype: float64



```

A scatter plot is also shown comparing actual vs predicted temperature values.

---

## Next Steps

- Add support for saving and loading trained models
- Experiment with other regressors (XGBoost, Gradient Boosting)
- Hyperparameter tuning
- Build a web interface using Streamlit or Flask
