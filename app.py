import streamlit as st
import joblib
import numpy as np

# Modeli yükleyin
sarima_model_fit = joblib.load('sarima_model.pkl')

# Streamlit başlığı
st.title("SARIMA Model Tahmin Uygulaması")

# Kullanıcıdan girdi al
st.subheader("Tahmin yapılacak dönem sayısını girin:")
n_periods = st.number_input("Dönem sayısı:", min_value=1, max_value=100, value=12)

# Tahmin düğmesi
if st.button("Tahmin Yap"):
    # Tahmin yap
    forecast = sarima_model_fit.forecast(steps=n_periods)
    
    # Tahmin sonuçlarını göster
    st.subheader("Tahmin Sonuçları:")
    st.write(forecast)

    # Tahmin sonuçlarını çiz
    st.line_chart(forecast)