import streamlit as st
import joblib

# Konfigurasi halaman
st.set_page_config(
    page_title="Residential Construction Cost Estimator",
    page_icon="🏠",
    layout="wide"
)

# Load model
model = joblib.load("lasso_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# Tampilan awal
st.title("🏠 Residential Construction Cost Estimator")

st.header("Property Information")

col1, col2 = st.columns(2)

with col1:
    building_area = st.number_input(
        "Building Area (m²)",
        min_value=37,
        max_value=500,
        value=120
    )

    land_area = st.number_input(
        "Land Area (m²)",
        min_value=46,
        max_value=1227,
        value=150
    )

    floors = st.number_input(
        "Floors",
        min_value=1,
        max_value=5,
        value=2
    )

with col2:
    bedrooms = st.number_input(
        "Bedrooms",
        min_value=1,
        max_value=10,
        value=3
    )

    bathrooms = st.number_input(
        "Bathrooms",
        min_value=1,
        max_value=10,
        value=2
    )

    carport = st.number_input(
        "Carport",
        min_value=0,
        max_value=5,
        value=1
    )

quality = st.selectbox(
    "Quality Level",
    ["Standard", "Medium", "Premium"]
)

location = st.selectbox(
    "Location Tier",
    ["Tier_1", "Tier_2", "Tier_3"]
)

if st.button("🔮 Predict Construction Cost"):

    # Encode Quality
    quality_medium = 1 if quality == "Medium" else 0
    quality_premium = 1 if quality == "Premium" else 0
    quality_standard = 1 if quality == "Standard" else 0

    # Encode Location
    location_tier2 = 1 if location == "Tier_2" else 0
    location_tier3 = 1 if location == "Tier_3" else 0

    # Susun data sesuai urutan feature_names
    input_data = [[
        building_area,
        land_area,
        floors,
        bedrooms,
        bathrooms,
        carport,
        quality_medium,
        quality_premium,
        quality_standard,
        location_tier2,
        location_tier3
    ]]

    # Scaling data
    scaled_data = scaler.transform(input_data)

    # Prediksi
    prediction = model.predict(scaled_data)

    # Tampilkan hasil
    st.divider()

    st.subheader("🏡 Prediction Result")

    cost_per_m2 = prediction[0] / building_area

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Estimated Construction Cost",
            f"Rp {prediction[0]:,.0f}"
        )

    with col2:
        st.metric(
            "Cost per m²",
            f"Rp {cost_per_m2:,.0f}"
        )

st.divider()

st.caption("""
**Disclaimer**

This application is a proof-of-concept developed to demonstrate how Machine Learning can support preliminary residential construction cost estimation.
The prediction model was trained using a **synthetic dataset** generated from realistic construction scenarios and Quantity Survey (QS) domain knowledge. Therefore, the estimated cost should be interpreted as an **initial reference**, not as a substitute for a detailed cost estimation.
For actual projects, a comprehensive Quantity Survey (QS) analysis and Bill of Quantities (BOQ) should be conducted before making budgeting or investment decisions.
""")