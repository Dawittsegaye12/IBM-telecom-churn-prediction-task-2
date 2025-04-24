import streamlit as st
import requests
import numpy as np

# Streamlit app title
st.title('Customer Churn Prediction')

# Input fields for user data
st.header('Enter Customer Details')
tenure = st.number_input('Tenure (in months)', min_value=0, max_value=100, value=12)
monthly_charges = st.number_input('Monthly Charges', min_value=0.0, max_value=500.0, value=50.0)
total_charges = st.number_input('Total Charges', min_value=0.0, max_value=10000.0, value=600.0)

# Collect features into a list
features = [tenure, monthly_charges, total_charges]

# Button to make prediction
if st.button('Predict Churn'):
    try:
        # Send data to Flask backend
        response = requests.post('http://127.0.0.1:5000/predict', json={'features': features})
        result = response.json()

        if 'error' in result:
            st.error(f"Error: {result['error']}")
        else:
            prediction = result['prediction']
            probability = result['probability']

            st.success(f"Prediction: {'Churn' if prediction == 1 else 'No Churn'}")
            st.info(f"Probability: {probability:.2f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")