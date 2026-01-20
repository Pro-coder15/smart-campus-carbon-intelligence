import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("campus_data.csv")

# Data cleaning
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)

# Carbon factors
ELECTRICITY_CF = 0.82
DIESEL_CF = 2.68
WATER_CF = 0.3 / 1000

# Carbon calculation
data["carbon_emission"] = (
    data["electricity_kwh"] * ELECTRICITY_CF +
    data["diesel_litres"] * DIESEL_CF +
    data["water_litres"] * WATER_CF
)

print("Carbon Emission Data:")
print(data[["date", "carbon_emission"]])

# AI Model
X = data[["electricity_kwh", "water_litres", "diesel_litres"]]
y = data["carbon_emission"]

model = LinearRegression()
model.fit(X, y)

future_usage = pd.DataFrame(
    [[1600, 9000, 70]],
    columns=["electricity_kwh", "water_litres", "diesel_litres"]
)

predicted_carbon = model.predict(future_usage)

predicted_carbon = model.predict(future_usage)

print("\nPredicted Carbon Emission for Next Day:")
print(f"{predicted_carbon[0]:.2f} kg CO2")

if predicted_carbon[0] > 1200:
    print("\nSuggestions:")
    print("- Reduce peak-hour electricity usage")
    print("- Optimize transport fuel usage")
    print("- Install solar panels")
