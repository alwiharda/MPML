import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("best_shoe_model.sav")

# Judul aplikasi
st.title("Prediksi Kualitas Sepatu Pria")

# Custom style
st.markdown("""
    <style>
    .main { background-color: #f4f4f4; }
    .stApp {
        background: linear-gradient(135deg, #dfe9f3, #ffffff);
        padding: 2rem;
        border-radius: 12px;
    }
    h1 {
        color: #2c3e50;
        font-family: 'Segoe UI', sans-serif;
        text-align: center;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSelectbox label, .stNumberInput label {
        font-weight: bold;
        color: #2c3e50;
    }
    .css-ffhzg2 {
        max-width: 700px;
        margin: auto;
    }
    </style>
""", unsafe_allow_html=True)

st.write("Masukkan informasi sepatu untuk memprediksi apakah rating >= 4 (layak beli).")

# Input pengguna
brand = st.selectbox("Pilih Brand", [
    "Bata", "Sparx", "ASIAN", "Kraasa", "Layasa", "World Wear Footwear", "Chevit",
    "Bersache", "Aqualite", "Centrino", "Zixer", "Shozie", "Birde", "T-Rock", "KNOOS"
])

price = st.number_input("Harga Saat Ini (₹)", min_value=100, max_value=10000, value=999)
sold = st.number_input("Jumlah Terjual", min_value=0, max_value=1000000, value=500)

# Buat DataFrame dari input pengguna
input_data = pd.DataFrame({
    'Brand_Name': [brand],
    'Current_Price': [price],
    'How_Many_Sold': [sold]
})

# Prediksi
if st.button("Prediksi"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("✅ Prediksi: Rating sepatu kemungkinan 4 atau lebih (rekomendasi bagus).")
    else:
        st.warning("⚠️ Prediksi: Rating sepatu kemungkinan di bawah 4.")
