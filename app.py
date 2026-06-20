# Tenure is in years not days

import streamlit as st
import pandas as pd
import numpy as np
import joblib

Kmeans = joblib.load('Kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Customer Segmentation App')

st.divider()

st.write('Enter customer details to predict the segment')

st.divider()

Age = st.number_input('Age', min_value=18, max_value=100)

Recency = st.number_input('Recency (days since last purchase)', min_value=0, max_value=365, value=30)

Income = st.number_input('Income', min_value=0, max_value=300000)

Num_Store_Purchases = st.number_input('Number of Store Purchases', min_value=0, max_value=200, value=10)

Num_Web_Purchases = st.number_input('Number of Web Purchases', min_value=0, max_value=200, value=10)

Num_Web_Visits_Month = st.number_input('Number of Web visits per Month', min_value=0, max_value=200, value=10)

Total_spending = st.number_input('Total_spending (sum of purchases)', min_value=0, max_value=5000, value=1000)

Customer_Tenure = st.number_input('Tenure', min_value=0, max_value=130, value=10)

st.divider()

input_data = pd.DataFrame({
    'Age': [Age],
    'Recency': [Recency],
    'Income': [Income],
    'NumStorePurchases': [Num_Store_Purchases],
    'NumWebPurchases': [Num_Web_Purchases],
    'NumWebVisitsMonth': [Num_Web_Visits_Month],
    'Total_spending': [Total_spending],
    'Customer_Tenure': [Customer_Tenure]
})
# SAME PREPROCESSING AS TRAINING NOTEBOOK
input_data["Income"] = np.log10(input_data["Income"] + 1)
input_data["NumStorePurchases"] = np.log10(input_data["NumStorePurchases"] + 1)
input_data["NumWebPurchases"] = np.log10(input_data["NumWebPurchases"] + 1)
input_data["Total_spending"] = np.log10(input_data["Total_spending"] + 1)

input_scaled = scaler.transform(input_data)

if st.button('Predict Segment'):
    cluster = Kmeans.predict(input_scaled)[0]
    st.success(f"Predicted Segment: Cluster {cluster}")
               
