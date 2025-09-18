import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Step 1: Load dataset
data = pd.read_csv("sales_data.csv")
data["Date"] = pd.to_datetime(data["Date"])
data.set_index("Date", inplace=True)

# Step 2: Visualize original sales
plt.figure(figsize=(10,5))
plt.plot(data.index, data["Sales"], marker="o", label="Actual Sales")
plt.title("Monthly Sales Data")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.show()

# Step 3: Build ARIMA model
model = ARIMA(data["Sales"], order=(2,1,2))  # p,d,q values
model_fit = model.fit()

# Step 4: Forecast next 6 months
forecast = model_fit.forecast(steps=6)
forecast_dates = pd.date_range(start=data.index[-1] + pd.offsets.MonthBegin(1), periods=6, freq="MS")

forecast_df = pd.DataFrame({"Date": forecast_dates, "Forecasted_Sales": forecast})
forecast_df.to_csv("forecast.csv", index=False)

# Step 5: Plot forecast
plt.figure(figsize=(10,5))
plt.plot(data.index, data["Sales"], marker="o", label="Actual Sales")
plt.plot(forecast_dates, forecast, marker="x", linestyle="--", color="red", label="Forecasted Sales")
plt.title("Sales Forecast (Next 6 Months)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.show()