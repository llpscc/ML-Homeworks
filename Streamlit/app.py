import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Загрузка модели
with open("streamlit_model.pkl", "rb") as f:
    bundle = pickle.load(f)

pipeline = bundle["pipeline"]
columns = bundle["columns"]
inverse_transform = bundle["inverse_transform"]
mean_values = bundle["mean_values"]

st.title("Предсказание стоимости автомобиля")
st.markdown("Введите значения признаков:")

user_input = {}
for col in columns:
    user_input[col] = st.number_input(col, value=float(mean_values[col]))

input_df = pd.DataFrame([user_input])

log_price = pipeline.predict(input_df)
predicted_price = inverse_transform(log_price)[0]

st.subheader("Предсказанная цена автомобиля:")
st.write(f"**{predicted_price:,.0f} ₽**")
