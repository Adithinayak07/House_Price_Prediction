import streamlit as st
import joblib
import pandas as pd
import os

st.title("🏠 House Price Predictor")

model_path = os.path.join(os.path.dirname(__file__), "..", "model.pkl")
model = joblib.load(model_path)
#model = joblib.load("model.pkl")

with st.form("predict"):
    rent = st.number_input("Monthly Rent", value=15000)
    area = st.number_input("Square Feet", value=1000)
    locality = st.text_input("Locality (e.g., Downtown)", "Downtown")
    BHK = st.number_input("Bedrooms", value=2, min_value=0, max_value=10, step=1)
    facing = st.text_input("Facing (e.g. , South)","South")
    parking = st.text_input("Parking (Open/Covered/Missing)", "Open")
    bathrooms = st.number_input("Bathrooms", value=2, min_value=1, max_value=10, step=1)
    submitted = st.form_submit_button("Predict")

#     locality = st.text_input("Locality (e.g., Downtown)", "Downtown")
#     #locality = ['Attibele','BTM Layout','Electronic City', 'Indiranagar','Jayanagar','K R Puram','Malleshwaram','Marathahalli','Yalahanka']
#     #selected_location = st.selectbox("Select Location:", locality)
#     BHK = st.number_input("Bedrooms", value=2, min_value=0, max_value=10, step=1)
#     parking = st.text_input("Parking (Open/Covered/Missing)", "Open")
#     #facing = ['North','South','East','West','North-east','North-west','South-east']
#     facing = st.text_input("Facing (e.g. , South)","South")
#     #facing = ['North','South','East','West','North-east','North-west','South-east']
#     #selected_facing = st.selectbox("Select Facing Direction:", facing)
#     bathrooms = st.number_input("Bathrooms", value=2, min_value=1, max_value=10, step=1)

if submitted:
    X = pd.DataFrame([{
        "rent": rent,
        "area": area,
        "locality": locality if locality else "Missing",
        "BHK":BHK,
        "facing": facing,
        "parking": parking if parking else "Missing",
        "bathrooms": bathrooms
    }])
    pred = model.predict(X)[0]
    st.success(f"Predicted Price: {pred:,.2f}")

#     locality = st.text_input("Locality (e.g., Downtown)", "Downtown")
#     #locality = ['Attibele','BTM Layout','Electronic City', 'Indiranagar','Jayanagar','K R Puram','Malleshwaram','Marathahalli','Yalahanka']
#     #selected_location = st.selectbox("Select Location:", locality)
#     BHK = st.number_input("Bedrooms", value=2, min_value=0, max_value=10, step=1)
#     parking = st.text_input("Parking (Open/Covered/Missing)", "Open")
#     #facing = ['North','South','East','West','North-east','North-west','South-east']
#     facing = st.text_input("Facing (e.g. , South)","South")
#     #facing = ['North','South','East','West','North-east','North-west','South-east']
#     #selected_facing = st.selectbox("Select Facing Direction:", facing)
#     bathrooms = st.number_input("Bathrooms", value=2, min_value=1, max_value=10, step=1)
#     submitted = st.form_submit_button("Predict")

# if submitted:
#     X = pd.DataFrame([{
#         "rent": rent,
#         "area": area,
#         "locality": locality if locality else "Missing",
#         "BHK": BHK,
#         "parking": parking if parking else "Missing",
#         "facing": facing,
#         "bathrooms": bathrooms
#     }])
#     pred = model.predict(X)[0]
#     st.success(f"Predicted Price: {pred:,.2f}")
