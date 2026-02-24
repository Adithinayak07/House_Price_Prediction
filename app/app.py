import streamlit as st
import requests


st.title("🏠 House Price Predictor")

API_URL = "http://api:8000/predict"   # IMPORTANT (service name, not localhost)

with st.form("predict"):
    area = st.number_input("Square Feet (area)", value=1000)
    rent = st.number_input("Rent", value=50000)
    facing = st.text_input("Facing Direction (e.g., East, North, West)", "East")
    locality = st.text_input("Locality (e.g., BTM Layout)", "BTM Layout")
    BHK = st.number_input("Bedrooms (BHK)", value=2, min_value=0, max_value=10, step=1)
    bathrooms = st.number_input("Bathrooms", value=2, min_value=0, max_value=10, step=1)
    parking = st.text_input("Parking (e.g., Bike, Car, Missing)", "Bike")

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = {
        "area": area,
        "rent": rent,
        "facing": facing if facing else "Missing",
        "locality": locality if locality else "Missing",
        "BHK": BHK,
        "bathrooms": bathrooms,
        "parking": parking if parking else "Missing"
    }

    try:
        response = requests.post(API_URL, json=input_data)

        if response.status_code == 200:
            prediction = response.json()["predicted_price_per_sqft"]
            st.success(f"Predicted Price per Sq Ft: {prediction:,.2f}")
        else:
            st.error("API Error")

    except Exception as e:
        st.error(f"Connection error: {e}")