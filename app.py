import joblib
import numpy as np
import streamlit as st
import pandas as pd
import os
model = joblib.load("results/solar_power_model.pkl")
# Page config
st.set_page_config(page_title="Solar Energy Analytics Dashboard", layout="wide")

# Title
st.title("☀️ Solar Energy Analytics & Predictive Maintenance Dashboard")

st.markdown("---")
st.write("This dashboard analyzes solar panel performance and predicts power output.")


# Load prediction dataset


data_path = "results/solar_prediction_results.csv"

if os.path.exists(data_path):

    df = pd.read_csv(data_path)

    st.subheader("Dataset Preview")

    st.dataframe(df.head())
    st.subheader("Dataset Statistics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(df))
    col2.metric("Average AC Power", round(df["AC_POWER"].mean(),2))
    col3.metric("Max AC Power", round(df["AC_POWER"].max(),2))

else:
    st.warning("⚠️ Prediction results file not found.")



# User Input Section


st.sidebar.title("Enter Weather Inputs")

irradiation = st.sidebar.slider("Irradiation", 0.0, 1.5, 0.8)
module_temp = st.sidebar.slider("Module Temperature", 0.0, 80.0, 35.0)
ambient_temp = st.sidebar.slider("Ambient Temperature", 0.0, 50.0, 25.0)
hour = st.sidebar.slider("Hour of Day", 0, 23, 12)


input_data = np.array([[irradiation, ambient_temp, module_temp, hour]])

predicted_power = model.predict(input_data)[0]
st.subheader("Predicted Solar Power Output")

st.metric("Predicted AC Power", f"{predicted_power:.2f}")

st.subheader("System Performance Overview")
expected_power = irradiation * 1000
efficiency = (predicted_power / (expected_power + 1)) * 100

col1, col2, col3 = st.columns(3)
col1.metric(
    label="Predicted AC Power",
    value=f"{predicted_power:.2f} kW"
)

col2.metric(
    label="Expected Power",
    value=f"{expected_power:.2f} kW"
)

col3.metric(
    label="System Efficiency",
    value=f"{efficiency:.2f}%"
)


st.subheader("Panel Maintenance Status")

if efficiency < 75 and irradiation > 0.6:
    st.error("⚠️ Panel Cleaning Recommended")
elif irradiation < 0.2:
    st.info("☁ Low solar irradiance detected")
else:
    st.success("✅ Panels Operating Normally")



# Display Plots


st.subheader("Model Analysis Plots")

plot_folder = "results"

plots = [
    "actual_vs_predicted_power.png",
    "irradiation_vs_power.png",
    "module_temperature_vs_power.png",
    "feature_importance.png"
]

for plot in plots:

    path = os.path.join(plot_folder, plot)

    if os.path.exists(path):
        st.image(path, caption=plot.replace("_"," ").title())


#prediction curve

st.subheader("Daily Power Prediction Curve")

hours = list(range(24))
predictions = []

for h in hours:
    input_data = np.array([[irradiation, ambient_temp, module_temp, h]])
    power = model.predict(input_data)[0]
    predictions.append(power)

prediction_df = pd.DataFrame({
    "Hour": hours,
    "Predicted Power": predictions
})
st.line_chart(
    prediction_df.set_index("Hour"),
    height=400
)


