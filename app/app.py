import streamlit as st
import joblib
import pandas as pd

st.title("🏠 House Price Predictor")

model = joblib.load("../model.pkl")

with st.form("predict"):
    area = st.number_input("Square Feet (area)", value=1000)
    rent=st.number_input("Rent",value=50000)
    facing = st.text_input("Facing Direction (e.g., East, North, West, etc.)", "East")
    locality = st.text_input("Locality (e.g., BTM Layout)", "BTM Layout")
    BHK = st.number_input("Bedrooms (BHK)", value=2, min_value=0, max_value=10, step=1)
    bathrooms = st.number_input("bathrooms", value=2, min_value=0, max_value=10, step=1)
    parking = st.text_input("Parking (e.g., Bike, Car, Bike and Car, Missing)", "Bike")
    submitted = st.form_submit_button("Predict")

if submitted:
    X = pd.DataFrame([{
        "area": area,
        "facing": facing if facing else "Missing",
        "locality": locality if locality else "Missing",
        "BHK": BHK,
        "bathrooms": bathrooms,
        "parking": parking if parking else "Missing"
    }])
    
    pred = model.predict(X)[0]
    st.success(f"Predicted Price: {pred:,.2f}")


