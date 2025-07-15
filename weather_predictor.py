import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import warnings

warnings.filterwarnings("ignore")


class WeatherPredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = [
            "humidity",
            "pressure",
            "wind_speed",
            "visibility",
            "apparent_temperature",
        ]
        self.target_column = "temperature"

    def load_data(self, data_path=None):
        if data_path and os.path.exists(data_path):
            print(f"Loading data from {data_path}")
            df = pd.read_csv(data_path)
            df.columns = [
                col.strip().lower().replace(" ", "_").replace("\n", "")
                for col in df.columns
            ]
            df.rename(
                columns={
                    "wind_bearing_(degrees)": "wind_bearing_degrees",
                    "loud_cover": "loud_cover",
                    "pressure": "pressure",
                },
                inplace=True,
            )
            df.dropna(subset=self.feature_columns + [self.target_column], inplace=True)
            return df
        else:
            raise FileNotFoundError("Data file not found!")

    def explore_data(self, df):
        print("\nExploratory Data Analysis")
        print("=" * 50)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nSummary Statistics:")
        print(df.describe().round(2))
        print("\nMissing values:")
        print(df.isnull().sum())
        numeric_df = df.select_dtypes(include=[np.number])
        if self.target_column in numeric_df.columns:
            correlations = numeric_df.corr()[self.target_column].sort_values(
                ascending=False
            )
            print("\nFeature Correlations with Temperature:")
            print(correlations.round(3))
        else:
            print("'temperature' column missing in numeric data.")
        return df

    def visualize_data(self, df):
        print("\nGenerating visualizations...")
        plt.style.use("seaborn-v0_8")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Weather Data Analysis", fontsize=16)
        axes[0, 0].hist(
            df["temperature"], bins=30, alpha=0.7, color="skyblue", edgecolor="black"
        )
        axes[0, 0].set_title("Temperature Distribution")
        correlation_matrix = df[self.feature_columns + [self.target_column]].corr()
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            ax=axes[0, 1],
            fmt=".2f",
        )
        axes[0, 1].set_title("Correlation Matrix")
        axes[0, 2].scatter(df["humidity"], df["temperature"], alpha=0.6, color="green")
        axes[0, 2].set_title("Humidity vs Temperature")
        axes[1, 0].scatter(df["pressure"], df["temperature"], alpha=0.6, color="red")
        axes[1, 0].set_title("Pressure vs Temperature")
        axes[1, 1].scatter(
            df["wind_speed"], df["temperature"], alpha=0.6, color="purple"
        )
        axes[1, 1].set_title("Wind Speed vs Temperature")
        axes[1, 2].scatter(
            df["visibility"], df["temperature"], alpha=0.6, color="orange"
        )
        axes[1, 2].set_title("Visibility vs Temperature")
        plt.tight_layout()
        plt.show()

    def preprocess_data(self, df):
        print("\nPreprocessing data...")
        df = df.dropna()
        for column in self.feature_columns + [self.target_column]:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]
        print(f"Data shape after preprocessing: {df.shape}")
        return df

    def split_data(self, df):
        print("\nSplitting data...")
        X = df[self.feature_columns]
        y = df[self.target_column]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, X_train, y_train):
        print("\nTraining model...")
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        self.model = rf
        print("Model training completed.")
        return self.model

    def evaluate_model(self, X_test, y_test):
        print("\nModel Evaluation")
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.3f}")
        return y_pred

    def visualize_predictions(self, y_test, y_pred):
        plt.figure(figsize=(10, 5))
        plt.scatter(y_test, y_pred, alpha=0.5, color="blue")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted Temperature")
        plt.show()


def main():
    predictor = WeatherPredictor()
    df = predictor.load_data("data/my_weather_data.csv")
    df = predictor.explore_data(df)
    predictor.visualize_data(df)
    df = predictor.preprocess_data(df)
    X_train, X_test, y_train, y_test = predictor.split_data(df)
    predictor.train_model(X_train, y_train)
    y_pred = predictor.evaluate_model(X_test, y_test)
    predictor.visualize_predictions(y_test, y_pred)


if __name__ == "__main__":
    main()
