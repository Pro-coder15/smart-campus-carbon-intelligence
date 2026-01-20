import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Page config
st.set_page_config(
    page_title="Smart Campus Carbon Intelligence",
    page_icon="🌱",
    layout="wide"
)

# Title
st.title("🌱 AI-Driven Smart Campus Carbon Intelligence System")
st.markdown("### Sustainable Infrastructure & SDG Goals")

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

# Sidebar
st.sidebar.header("⚙️ Simulation Controls")
future_electricity = st.sidebar.slider("Electricity Usage (kWh)", 1000, 2000, 1600)
future_water = st.sidebar.slider("Water Usage (Litres)", 7000, 10000, 9000)
future_diesel = st.sidebar.slider("Diesel Usage (Litres)", 40, 100, 70)

# Train AI model
X = data[["electricity_kwh", "water_litres", "diesel_litres"]]
y = data["carbon_emission"]

model = LinearRegression()
model.fit(X, y)

future_input = pd.DataFrame(
    [[future_electricity, future_water, future_diesel]],
    columns=["electricity_kwh", "water_litres", "diesel_litres"]
)

predicted_carbon = model.predict(future_input)[0]

# Metrics row
col1, col2, col3 = st.columns(3)
col1.metric("⚡ Avg Electricity (kWh)", int(data["electricity_kwh"].mean()))
col2.metric("💧 Avg Water (Litres)", int(data["water_litres"].mean()))
col3.metric("🛢 Avg Diesel (Litres)", int(data["diesel_litres"].mean()))

st.divider()

# Charts
st.subheader("📊 Campus Carbon Emission Trend")

fig, ax = plt.subplots()
ax.plot(data["date"], data["carbon_emission"], marker="o")
ax.set_xlabel("Date")
ax.set_ylabel("Carbon Emission (kg CO₂)")
plt.xticks(rotation=45)
st.pyplot(fig)

# Prediction section
st.subheader("🔮 AI Prediction Result")
st.metric("Predicted Next-Day Carbon Emission (kg CO₂)", f"{predicted_carbon:.2f}")

# Suggestions
st.subheader("💡 AI Sustainability Suggestions")

if predicted_carbon > 1400:
    st.error("⚠️ High Carbon Emission Detected")
    st.write("- Reduce peak-hour electricity usage")
    st.write("- Optimize college bus routes")
    st.write("- Install solar panels on hostels")
elif predicted_carbon > 1200:
    st.warning("⚠️ Moderate Carbon Emission")
    st.write("- Monitor energy usage closely")
    st.write("- Promote energy-efficient appliances")
else:
    st.success("✅ Carbon Emission Under Control")
    st.write("- Maintain current sustainability practices")

# SDG Mapping
st.divider()
st.subheader("🌍 SDG Alignment")

st.write("""
- **SDG 7:** Affordable and Clean Energy  
- **SDG 11:** Sustainable Cities and Communities  
- **SDG 12:** Responsible Consumption  
- **SDG 13:** Climate Action  
""")

# Footer
st.caption("🔌 IoT-Ready | 🤖 AI-Powered | 🌱 Sustainable Campus")
