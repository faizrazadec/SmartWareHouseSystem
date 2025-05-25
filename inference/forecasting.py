import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from datetime import timedelta

MODEL_PATH = '/home/faizraza/Personal/Projects/SmartWareHouseSystem/models/forecasting_model/Farah_model5.h5'
SCALER_PATH = '/home/faizraza/Personal/Projects/SmartWareHouseSystem/models/forecasting_model/minmax_scaler.pkl'
TIME_STEPS = 30
DAYS_TO_FORECAST = 30
PRODUCTS_TO_PLOT = ['Product_1263', 'Product_0374', 'Product_0349', 'Product_0033', 'Product_1341']
DATA_PATH = '/home/faizraza/Personal/Projects/SmartWareHouseSystem/models/forecasting_model/data_resampled_daily.pkl'
# --------------------------------------------

def load_scaler_and_model():
    scaler = joblib.load(SCALER_PATH)
    model = load_model(MODEL_PATH)
    return scaler, model

def create_forecast_input(scaled_data, time_steps=30):
    return scaled_data[-time_steps:]

def forecast_future(model, input_seq, time_steps, forecast_days):
    forecasted = []
    current_seq = input_seq.copy()

    for _ in range(forecast_days):
        pred = model.predict(current_seq.reshape(1, time_steps, current_seq.shape[1]), verbose=0)
        forecasted.append(pred[0])
        current_seq = np.append(current_seq[1:], [pred[0]], axis=0)

    return np.array(forecasted)

def run_inference(data_resampled_daily, forecast_days=DAYS_TO_FORECAST):
    scaler, model = load_scaler_and_model()

    # Scale data
    scaled_data = scaler.transform(data_resampled_daily)

    # Prepare input for forecasting
    last_input = create_forecast_input(scaled_data, time_steps=TIME_STEPS)

    # Forecast
    forecast_scaled = forecast_future(model, last_input, TIME_STEPS, forecast_days)

    # Inverse scale
    forecast = scaler.inverse_transform(forecast_scaled)

    # Prepare output DataFrame
    forecast_dates = pd.date_range(start=data_resampled_daily.index[-1] + timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame(data=forecast, index=forecast_dates, columns=data_resampled_daily.columns)

    return forecast_df

def plot_forecast_vs_actual(actual_df, forecast_df, selected_products=PRODUCTS_TO_PLOT):
    plt.figure(figsize=(14, 6))
    for product in selected_products:
        if product in actual_df.columns and product in forecast_df.columns:
            plt.plot(actual_df.index, actual_df[product], label=f'{product} (Last 30 days)', linestyle='--')
            plt.plot(forecast_df.index, forecast_df[product], label=f'{product} (Forecast)', linewidth=2)

    plt.title("Last 30 Days vs Forecasted Demand (Next 30 Days)")
    plt.xlabel("Date")
    plt.ylabel("Order Demand")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load historical resampled daily data
    historical_df = pd.read_pickle(DATA_PATH)

    # Extract last 30 days of actual data
    actual_last_30 = historical_df[-30:]

    # Run inference to forecast next 30 days
    forecast_df = run_inference(historical_df)

    # Plot both actual and forecast
    plot_forecast_vs_actual(actual_last_30, forecast_df)

    # Optional: Save forecast to CSV
    forecast_df.to_csv("forecast_next_30_days.csv")
