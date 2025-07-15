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

## Example Output

```
Loading data from data/my_weather_data.csv
Exploratory Data Analysis
Shape: (7895, 10)
MAE: 1.24 | RMSE: 2.15 | R2: 0.876
```

A scatter plot is also shown comparing actual vs predicted temperature values.

---

## Next Steps

- Add support for saving and loading trained models
- Experiment with other regressors (XGBoost, Gradient Boosting)
- Hyperparameter tuning
- Build a web interface using Streamlit or Flask
