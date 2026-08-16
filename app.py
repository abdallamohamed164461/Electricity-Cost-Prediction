import streamlit as st
import pandas as pd
import numpy as np
import joblib
 
st.set_page_config(page_title="Electricity Cost Predictor", page_icon="⚡", layout="centered")
 
 
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, feature_columns
 
model, scaler, feature_columns = load_artifacts()
 
STRUCTURE_TYPES = ["Residential", "Commercial", "Mixed-use", "Industrial"]
 
st.title("⚡ Electricity Cost Predictor")
st.write(
    "This app predicts the **electricity cost** of a site based on its "
    "structural, resource-usage, and environmental characteristics."
)
 
st.divider()
 
st.subheader("Site Information")
 
col1, col2 = st.columns(2)
 
with col1:
    site_area = st.slider(
        "Site Area",
        min_value=100, max_value=5000, value=2500, step=10,
        help="Total area of the site/building. Larger sites generally consume more electricity.",
        key="site_area",
    )
    water_consumption = st.slider(
        "Water Consumption",
        min_value=200, max_value=10900, value=3500, step=50,
        help="Total water usage of the site. Often correlates with occupancy and building activity.",
        key="water_consumption",
    )
    recycling_rate = st.slider(
        "Recycling Rate (%)",
        min_value=10, max_value=90, value=50, step=1,
        help="Percentage of waste that is recycled at the site.",
        key="recycling_rate",
    )
    resident_count = st.slider(
        "Number of Residents",
        min_value=0, max_value=490, value=85, step=1,
        help="Number of people living/working at the site. More residents usually means higher electricity demand.",
        key="resident_count",
    )
 
with col2:
    utilisation_rate = st.slider(
        "Space Utilisation Rate (%)",
        min_value=30, max_value=100, value=65, step=1,
        help="How much of the site's capacity is actively being used (e.g. occupied units, active floors).",
        key="utilisation_rate",
    )
    air_quality_index = st.slider(
        "Air Quality Index",
        min_value=0, max_value=200, value=100, step=1,
        help="Air quality around the site. 0 = best air quality, 200 = worst (most polluted).",
        key="air_quality_index",
    )
    issue_resolution_time = st.slider(
        "Avg. Issue Resolution Time (days)",
        min_value=1, max_value=72, value=15, step=1,
        help="Average number of days it takes to resolve a maintenance/facility issue at the site.",
        key="issue_resolution_time",
    )
    structure_type = st.selectbox(
        "Building Type",
        STRUCTURE_TYPES,
        help="The category of the building: Residential, Commercial, Industrial, or Mixed-use.",
        key="structure_type",
    )
 
st.divider()
 

if st.button("🔮 Predict Electricity Cost", use_container_width=True, key="predict_button"):
 
    input_dict = {
        "site area": site_area,
        "water consumption": water_consumption,
        "recycling rate": recycling_rate,
        "utilisation rate": utilisation_rate,
        "air qality index": air_quality_index,
        "issue reolution time": issue_resolution_time,
        "resident count": resident_count,
        "structure type": structure_type,
    }
    input_df = pd.DataFrame([input_dict])
 
    input_encoded = pd.get_dummies(input_df, columns=["structure type"], drop_first=True)
 
    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[feature_columns]
 
    input_scaled = scaler.transform(input_encoded)
     
    
    prediction = model.predict(input_scaled)[0]
 
    st.success(f"### Estimated Electricity Cost: **{prediction:,.2f}** ")
    st.caption(
        "This is a point estimate from the trained model. Actual costs may vary "
        "based on factors not captured in this dataset."
    )
 
st.divider()
st.caption("Built with Streamlit • Model trained on the Electricity Cost Prediction Dataset")
