import streamlit as st
import numpy as np
import pickle

# ---------------------------------------------------------
# Load Pickle Model
# ---------------------------------------------------------
model = pickle.load(open("model.pkl", "rb"))

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.title("Medical Charges Prediction")
st.write("Predict charges based on **Age** and **Smoking Status**")

# Input: Age
age = st.number_input("Enter Age", min_value=1, max_value=100, value=25)

# Input: Smoker or Not
smoker = st.selectbox("Do you smoke?", ["no", "yes"])

# Convert smoker to numeric
smoker_value = 1 if smoker == "yes" else 0

# Predict Button
if st.button("Predict Charges"):
    input_data = np.array([[age, smoker_value]])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Medical Charges: ₹{prediction:,.2f}")
