# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from wildfire_system import WildfireDataSystem  # <-- Use your class in wildfire_system.py

# ---------------------------------------------
# Load or train the model once
# ---------------------------------------------
system = WildfireDataSystem()

if not system.model:
    try:
        system.load_model()
        st.sidebar.success("✅ Model loaded")
    except:
        st.sidebar.warning("🔄 Training new model...")
        system.train()
        st.sidebar.success("✅ Model trained")

# ---------------------------------------------
# Streamlit UI
# ---------------------------------------------
st.title("🔥 Wildfire Risk Predictor")
st.write(
    """
    Enter environmental factors below to estimate the risk level for wildfire occurrence.
    """
)

# 🌍 Location & date
col1, col2 = st.columns(2)
latitude = col1.number_input("Latitude", value=34.5)
longitude = col2.number_input("Longitude", value=-118.5)

date = st.date_input("Date", value=datetime.now().date())

# 🌡️ Weather & Landcover
temperature = st.slider("Temperature (°C)", min_value=-10, max_value=50, value=30)
humidity = st.slider("Humidity (%)", min_value=0, max_value=100, value=40)
precipitation = st.slider("Precipitation (mm)", min_value=0, max_value=50, value=0)
wind_speed = st.slider("Wind Speed (km/h)", min_value=0, max_value=100, value=20)
landcover_type = st.selectbox(
    "Landcover Type",
    options=[10, 20, 30, 40, 50, 60],
    index=2
)
elevation = st.number_input("Elevation (m)", min_value=0, max_value=5000, value=500)
ndvi = st.slider("NDVI (0 to 1)", min_value=0.0, max_value=1.0, value=0.6)

# ---------------------------------------------
# Predict Button
# ---------------------------------------------
if st.button("Predict Wildfire Risk"):
    result = system.predict_risk({
        'latitude': latitude,
        'longitude': longitude,
        'temperature': temperature,
        'humidity': humidity,
        'precipitation': precipitation,
        'wind_speed': wind_speed,
        'landcover_type': landcover_type,
        'month': pd.to_datetime(date).month,
        'day': pd.to_datetime(date).day,
        'elevation': elevation,
        'ndvi': ndvi
    })
    st.subheader("🌲 Wildfire Risk Prediction")
    st.write(f"**Probability:** {result['probability']:.1%}")
    st.write(f"**Risk Level:** {result['risk_level']}")
    st.json(result['features'])

# ---------------------------------------------
# Footer
# ---------------------------------------------
st.markdown(
    """
    ---
    ✅ **Powered by Random Forest ML**
    """
)
