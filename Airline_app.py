import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import os

# Page config
st.set_page_config(page_title="Flight Fare Predictor BDT", page_icon="✈️", layout="wide")

# Sidebar
st.sidebar.title("📌 Project Overview")
st.sidebar.write("""
This project predicts international Airline Flight Fares from Bangladesh using a machine learning model.
Developed as part of Dursikshya Data Science Certificate Project by Balram Shah.
""")
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Key Visual Insights")
st.sidebar.markdown("""
- 📊 Avg Fare by Airline
- 🛫 Route Popularity
- ⏰ Time vs Fare
- 🏷️ Seasonality Impact
- 🎟️ Booking & Class Trends
- ✈️ Flight Type Comparison
""")

# Header
st.markdown("""
<h1 style='text-align:center; color:#1E90FF;'>✈️ Flight Fare Predictor BD</h1>
<h4 style='text-align:center;'>Project by Balram Shah | Dursikshya 2025</h4>
""", unsafe_allow_html=True)


# Charts Section
st.markdown("### 📈 Show EDA Charts")

charts = [
    ("Average_Fare_By_Top_Airlines.png", "Average Fare By Top Airlines"),
    ("Average_Fare_By_Top_Routes.png", "Average Fare By Top Routes"),
    ("Average_Fare_By_Departure_Hour.png", "Average Fare By Departure Hour"),
    ("Average_Fare_by_Season.png", "Average Fare by Season"),
    ("Base_Fare_vs_Tax_&_Surcharge_by_class.png", "BaseFare vs Tax & Surcharge by class"),
    ("Fare_Distribution_By_Airline_Category.png", "Fare Distribution By Airline Category"),
    ("Fare_Disribution_Direct_vs_Stopover.png", "Fare Distribution Direct vs Stopover"),
    ("Flight_Share_BY_Class.png", "Flight Share BY Class")
]

for file, caption in charts:
    with st.expander(f"Click to view: {caption}"):
        if os.path.exists(file):
            st.image(file, caption=caption)
        else:
            st.warning(f"⚠️ '{file}' not found. Please upload it to display this chart.")


# Load model and scaler
model = joblib.load("Airline_rf_model.joblib")
scaler = joblib.load("Airline_Flight_Fare_Scaler.joblib")


# Mappings
airline_map = {'Air Arabia': 0, 'Air Astra': 1, 'Air India': 2, 'AirAsia': 3,
               'Biman Bangladesh Airlines': 4, 'British Airways': 5, 'Cathay Pacific': 6,
               'Emirates': 7, 'Etihad Airways': 8, 'FlyDubai': 9, 'Gulf Air': 10,
               'IndiGo': 11, 'Kuwait Airways': 12, 'Lufthansa': 13, 'Malaysian Airlines': 14,
               'NovoAir': 15, 'Qatar Airways': 16, 'Saudia': 17, 'Singapore Airlines': 18,
               'SriLankan Airlines': 19, 'Thai Airways': 20, 'Turkish Airlines': 21,
               'US-Bangla Airlines': 22, 'Vistara': 23}

source_map = {'BZL': 0, 'CGP': 1, 'CXB': 2, 'DAC': 3, 'JSR': 4, 'RJH': 5, 'SPD': 6, 'ZYL': 7}

dest_map = {'BKK': 0, 'BZL': 1, 'CCU': 2, 'CGP': 3, 'CXB': 4, 'DAC': 5, 'DEL': 6, 'DOH': 7,
            'DXB': 8, 'IST': 9, 'JED': 10, 'JFK': 11, 'JSR': 12, 'KUL': 13, 'LHR': 14,
            'RJH': 15, 'SIN': 16, 'SPD': 17, 'YYZ': 18, 'ZYL': 19}

season_map = {'Eid': 0, 'Hajj': 1, 'Regular': 2, 'Winter Holidays': 3}

aircraft_map = {'Airbus A320': 0, 'Airbus A350': 1, 'Boeing 737': 2, 'Boeing 777': 3, 'Boeing 787': 4}

booking_map = {'Direct Booking': 0, 'Online Website': 1, 'Travel Agency': 2}

class_map = {'Business': 0, 'Economy': 1, 'First Class': 2}

# User inputs
st.markdown("### ✍️ Enter Your Flight Details")
columns = st.columns(3)

with columns[0]:
    base_fare = st.number_input("Base Fare (BDT)", min_value=0.0)
    tax = st.number_input("Tax & Surcharge (BDT)", min_value=0.0)
    days_before = st.number_input("Days Before Departure", min_value=0)
    flight_mins = st.number_input("Flight Duration (mins)", min_value=0.0)
    dep_hour = st.number_input("Departure Hour", min_value=0, max_value=23)
    dep_min = st.number_input("Departure Minute", min_value=0, max_value=59)

with columns[1]:
    holiday_flag = st.selectbox("Holiday Flag", [0, 1])
    is_premium = st.selectbox("Is Premium Airline", [0, 1])
    is_direct = st.selectbox("Is Direct", [0, 1])
    is_night = st.selectbox("Is Night Flight", [0, 1])
    dep_month = st.number_input("Departure Month", min_value=1, max_value=12)
    airline = st.selectbox("Airline", list(airline_map.keys()))

with columns[2]:  
    source = st.selectbox("Source", list(source_map.keys()))
    dest = st.selectbox("Destination", list(dest_map.keys()))
    season = st.selectbox("Seasonality", list(season_map.keys()))
    aircraft = st.selectbox("Aircraft", list(aircraft_map.keys()))
    booking = st.selectbox("Booking Type", list(booking_map.keys()))
    cls = st.selectbox("Class", list(class_map.keys()))

# Predict button
if st.button("🎯 Predict Fare"):
    input_df = pd.DataFrame([{
        'Base Fare (BDT)': base_fare,
        'Tax & Surcharge (BDT)': tax,
        'Days Before Departure': days_before,
        'flight_mins': flight_mins,
        'holiday_flag': holiday_flag,
        'Is_Premium_Airline': is_premium,
        'Is_Direct': is_direct,
        'Is_Night_Flight': is_night,
        'Departure_Hour': dep_hour,
        'Departure_Minute': dep_min,
        'Departure_Month': dep_month,
        'Airline_Label': airline_map[airline],
        'Source_Label': source_map[source],
        'Destination_Label': dest_map[dest],
        'Seasonality_Label': season_map[season],
        'Aircraft_Label': aircraft_map[aircraft],
        'Booking_Label': booking_map[booking],
        'Class_Label': class_map[cls]
    }])

    st.markdown("### 📋 Input Data")
    st.dataframe(input_df)

    try:
        scaled_input = scaler.transform(input_df)
        log_prediction = model.predict(scaled_input)[0]
        final_fare = np.expm1(log_prediction)
        st.success(f"💰 Predicted Fare: {final_fare:.2f} BDT")

        fig = px.bar(x=["Predicted Fare"], y=[final_fare],
                     labels={"x": "", "y": "Fare (BDT)"},
                     title="📊 Predicted Fare Bar Chart", text=[f"{final_fare:.2f}"],
                     color_discrete_sequence=["#1E90FF"])
        
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig)

        st.download_button("📥 Download Prediction CSV", input_df.to_csv(index=False).encode(), "prediction.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error: {e}")


# Footer
st.markdown("""
    <hr style="height:2px;border:none;background-color:#666;" />
    <p style="text-align:center;font-size:13px;">© 2025 Balram Shah • Flight Fare Prediction App | Dursikshya Internship</p>
""", unsafe_allow_html=True)