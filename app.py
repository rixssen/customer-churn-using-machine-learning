import joblib
import streamlit as st
import numpy as np
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("churn prediction app")
st.divider()
st.write("please enter the values and hit the predict button for getting a prediction.")

st.divider()

age = st.number_input("enter age" ,min_value=10 ,max_value= 100 , value= 100)

tenure = st.number_input("enter tenure", min_value=0 , max_value= 100 , value= 10)

montlycharge = st.number_input("enter montly charge" , min_value=30 , max_value=150)

gender = st.selectbox("enter the gender", ["male","female"])

st.divider()

predictbutton = st.button("predict")

if predictbutton:

    gender_selected = 1 if gender == "Female" else 0

    x=[age , gender_selected , tenure , montlycharge]

    X1= np.array(x)

    x_array = scaler.transform([X1])

    prediction = model.predict(x_array)[0]

    predicted = "yes churn" if prediction ==1 else "no churn"

    st.write(f"predict: {predicted}")
else:
    
    st.write("please enter the values and use predict button")