import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Загружаем модель
with open("streamlit_model.pkl", "rb") as f:
    bundle = pickle.load(f)

pipeline = bundle['pipeline']
inverse_transform = bundle['inverse_transform']
columns = bundle['columns']
mean_values = bundle['mean_values']

st.title("Предсказание цены автомобиля")

user_input = {}
for col in columns:
    user_input[col] = st.number_input(col, value=float(mean_values[col]))

input_df = pd.DataFrame([user_input])

log_pred = pipeline.predict(input_df)
price = inverse_transform(log_pred)[0]

st.subheader("Предсказанная цена:")
st.write(f"{price:,.0f} руб.")
