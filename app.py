import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

@st.cache_resource
def train_model():
    # Load dataset
    df = pd.read_csv("MEN_SHOES.csv")

    # Preprocessing kolom numerik dan kategorik
    df['How_Many_Sold'] = df['How_Many_Sold'].replace(',', '', regex=True)
    df['How_Many_Sold'] = pd.to_numeric(df['How_Many_Sold'], errors='coerce')

    df['Current_Price'] = df['Current_Price'].replace('[₹,]', '', regex=True)
    df['Current_Price'] = pd.to_numeric(df['Current_Price'], errors='coerce')

    df['RATING'] = pd.to_numeric(df['RATING'], errors='coerce')

    df.dropna(inplace=True)


    # Target dan fitur
    X = df[['Brand_Name', 'Current_Price', 'How_Many_Sold']]
    y = df['RATING'].apply(lambda x: 'high' if x >= 4.0 else 'low')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pipeline preprocessing + training
    numeric_features = ['Current_Price', 'How_Many_Sold']
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_features = ['Brand_Name']
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    clf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    clf.fit(X_train, y_train)

    return clf

# Train model hanya sekali
model = train_model()

# --- UI Streamlit ---
st.set_page_config(layout="centered")
st.title("🎯 Prediksi Kualitas Sepatu Pria")
st.write("Masukkan informasi sepatu di bawah untuk memprediksi apakah rating-nya tinggi atau rendah.")

brand = st.selectbox("Merek Sepatu", ['Nike', 'Adidas', 'Puma', 'Bata', 'Reebok', 'Vans', 'Woodland'])  # tambah sesuai data kamu
price = st.number_input("Harga Saat Ini (₹)", min_value=100, step=50)
sold = st.number_input("Jumlah Terjual", min_value=0, step=10)

if st.button("Prediksi"):
    data = pd.DataFrame({
        'Brand_Name': [brand],
        'Current_Price': [price],
        'How_Many_Sold': [sold]
    })

    prediction = model.predict(data)[0]
    st.subheader("📌 Hasil Prediksi:")
    st.success(f"Rating Sepatu Diprediksi: **{prediction.upper()}**")

# Optional styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(to bottom right, #f0f2f6, #ffffff);
    }
</style>
""", unsafe_allow_html=True)



