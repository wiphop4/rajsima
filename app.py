import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# โหลดโมเดล PINNs ที่ฝึกเสร็จแล้ว
model = load_model('pinns_pm25_model')

# โหลด scaler ที่เซฟไว้
with open('scalers.pkl', 'rb') as file:
    scalers = pickle.load(file)

# ฟังก์ชั่นสำหรับการแปลงค่าด้วย scaler
def scale_input(lat, lon, timestamp):
    lat_scaled = scalers['latitude'].transform([[lat]])[0][0]
    lon_scaled = scalers['longitude'].transform([[lon]])[0][0]
    time_scaled = scalers['time'].transform([[timestamp]])[0][0]
    return np.array([lat_scaled, lon_scaled, time_scaled])

# ฟังก์ชั่นสำหรับการทำนาย PM2.5
def predict_pm25(lat, lon, timestamp):
    # แปลงข้อมูล input ให้เป็นค่าที่ scaled
    scaled_input = scale_input(lat, lon, timestamp)
    
    # ทำนายค่า PM2.5 จากโมเดล
    pm25_scaled = model.predict(scaled_input.reshape(1, -1))
    
    # แปลงค่ากลับเป็น PM2.5 ที่ไม่ scaled
    pm25 = scalers['pm25'].inverse_transform(pm25_scaled)
    
    return pm25[0][0]

# ใช้ CSS เพื่อปรับแต่ง UI ให้ทันสมัย
st.markdown("""
    <style>
    .main {
        font-family: 'Arial', sans-serif;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 15px 32px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .stTextInput input, .stNumberInput input {
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
    }
    .stTitle {
        text-align: center;
        color: #3e8e41;
    }
    </style>
""", unsafe_allow_html=True)

# สร้าง UI ด้วย Streamlit
st.title('PM2.5 Prediction with PINNs', anchor="top")
st.write('กรอกข้อมูล latitude, longitude และ Unix timestamp เพื่อทำนายค่าฝุ่น PM2.5')

# รับข้อมูลจากผู้ใช้
latitude = st.number_input('Latitude', min_value=-90.0, max_value=90.0, value=13.7563, step=0.0001)
longitude = st.number_input('Longitude', min_value=-180.0, max_value=180.0, value=100.5018, step=0.0001)
timestamp = st.number_input('Unix Timestamp', min_value=0, max_value=9999999999, value=1622548800)

# กดปุ่มเพื่อทำนาย
if st.button('ทำนายค่า PM2.5'):
    pm25 = predict_pm25(latitude, longitude, timestamp)
    st.write(f'ค่าฝุ่น PM2.5 ที่ตำแหน่ง ({latitude}, {longitude}) และเวลา {timestamp} คือ: {pm25:.2f} µg/m³', 
             unsafe_allow_html=True)

