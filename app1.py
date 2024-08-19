import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Modeli yükleyin
sarima_model_fit = joblib.load('sarima_model.pkl')

# Streamlit başlığı
st.title("SARIMA Model Tahmin Uygulaması")

# Kullanıcıdan başlangıç ayı ve yılı al
st.subheader("Tahmin için başlangıç ayını ve yılını seçin:")
start_year = st.selectbox("Yıl:", list(range(2024, 2031)), index=24)  # 2024 varsayılan olarak seçili
start_month = st.selectbox("Ay:", list(range(1, 13)), index=7)  # Ağustos varsayılan olarak seçili

# Kullanıcıdan girdi al
st.subheader("Tahmin yapılacak dönem sayısını girin:")
n_periods = st.number_input("Dönem sayısı (ay cinsinden):", min_value=1, max_value=100, value=12)

# Tahmin düğmesi
if st.button("Tahmin Yap"):
    # Tahmin yap
    forecast = sarima_model_fit.forecast(steps=n_periods)
    
    # Tahmin sonuçlarını tarihlerle ilişkilendir
    start_date = datetime(start_year, start_month, 1)
    date_range = pd.date_range(start=start_date, periods=n_periods, freq='M')
    forecast_df = pd.DataFrame({'Tarih': date_range, 'Tahmini Gelir': forecast})
    
    # Tahmin sonuçlarını göster
    st.subheader("Tahmin Sonuçları:")
    st.write(forecast_df)
    
    # Tahmin sonuçlarını çiz
    st.line_chart(forecast_df.set_index('Tarih')['Tahmini Gelir'])
