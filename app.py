import streamlit as st
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

## Load the model and the scaler :
with  open('iris_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

with open('iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Iris Data Prediction App")
st.write("Enter the 4 features of your flower for predicting the flower name")

sepal_length = st.number_input("Enter the sepal length", min_value=0.0)
sepal_width = st.number_input("Enter the sepal width", min_value=0.0)
petal_length = st.number_input("Enter the petal length", min_value=0.0)
petal_width = st.number_input("Enter the petal width", min_value=0.0)

if st.button("Predict"):
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    scaled_features = scaler.transform(input_features)
    prediction = model.predict(scaled_features)
    flower_names = ["Setosa", "Versicolor", "Virginica"]
    st.success("The flower name is {}".format(flower_names[prediction[0]]))
