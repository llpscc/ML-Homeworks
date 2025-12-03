import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge

# Загрузка модели
with open("streamlit_model.pkl", "rb") as f:
    bundle = pickle.load(f)

pipeline = bundle["pipeline"]
inverse_transform = bundle["inverse_transform"]

input_features = pipeline.named_steps['preprocessor'].transformers_[0][2] + \
                 pipeline.named_steps['preprocessor'].transformers_[1][2]

# Интерфейс
st.title("Предсказание стоимости автомобиля")
st.markdown("Введите характеристики авто:")

user_input = {}

for col in input_features:
    user_input[col] = st.text_input(col) if col.isalpha() else st.number_input(col)

input_df = pd.DataFrame([user_input])

# Предсказание
try:
    log_price = pipeline.predict(input_df)
    predicted_price = inverse_transform(log_price)[0]

    st.subheader("Предсказанная цена:")
    st.write(f"**{predicted_price:,.0f} ₽**")
except Exception as e:
    st.error(f"Ошибка: {e}")
