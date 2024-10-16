import streamlit as st
import numpy as np
from joblib import load
from PIL import Image

model = load('data/iris_model.joblib')

st.title('Iris Flower Type Predictor')

col1, col2 =st.columns (2)

with col1: 
    
    sepal_len = st.number_input("Sepal length (cm):", min_value=0.0, max_value=15.0)
    sepal_width = st.number_input("Sepal width (cm):", min_value=0.0, max_value=15.0)

with col2:
    petal_len = st.number_input("Petal length (cm):", min_value=0.0, max_value=15.0)
    petal_width = st.number_input("Petal width (cm):", min_value=0.0, max_value=15.0)


def make_pred(col1, col2):
    input_array = np.array([sepal_l, sepal_w, petal_l, petal_w]).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    return prediction

if st.button('Predict Flower Type'):
    prediction = make_pred(sepal_len, sepal_width, petal_len, petal_width)
    
    if prediction == 0:
        st.write("Predicted Flower: Iris Setosa")
        image = Image.open('images/iris_setosa.png')
    elif prediction == 1:
        st.write("Predicted Flower: Iris Versicolor")
        image = Image.open('images/iris_versicolor.png')
    else:
        st.write("Predicted Flower: Iris Virginica")
        image = Image.open('images/iris_virginica.png')

    st.image(image, use_column_width=True)


