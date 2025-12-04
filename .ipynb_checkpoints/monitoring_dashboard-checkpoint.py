# monitoring_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
from production_clv_system import ProductionCLVSystem, CLVDashboard

def create_monitoring_dashboard():
    """
    Create interactive monitoring dashboard
    """
    st.title("Customer Lifetime Value Monitoring Dashboard")
    
    # Load latest segmented data
    clv_system = ProductionCLVSystem(model_path="production_models/")
    clv_system.load_models()
    
    # This would load actual data in production
    # For demo purposes, we'll simulate the data structure
    segment_data = pd.DataFrame({
        'segment': ['Champions', 'Loyal Customers', 'Potential Loyalists', 'At Risk'],
        'customer_count': [1500, 2500, 3500, 2500],
        'avg_clv': [1842, 687, 294, 87],
        'avg_frequency': [8.2, 4.1, 2.3, 1.2],
        'avg_recency': [45, 90, 180, 365]
    })
    
    # Customer segment distribution
    st.subheader("Customer Segment Distribution")
    fig1 = px.pie(segment_data, values='customer_count', names='segment', 
                  title="Customer Segments by Count")
    st.plotly_chart(fig1)
    
    # Average CLV by segment
    st.subheader("Average CLV by Segment")
    fig2 = px.bar(segment_data, x='segment', y='avg_clv', 
                  title="Average CLV by Customer Segment")
    st.plotly_chart(fig2)
    
    # Performance metrics (would be real data in production)
    st.subheader("Model Performance Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Correlation", "0.85", "+0.02")
    col2.metric("MAE", "$76.32", "-$5.21")
    col3.metric("Customers", "10,000", "+245")
    
    # Real-time prediction interface
    st.subheader("Real-time CLV Prediction")
    with st.form("clv_prediction"):
        customer_id = st.text_input("Customer ID")
        recency = st.number_input("Recency (days)", min_value=0, value=30)
        frequency = st.number_input("Frequency", min_value=1, value=5)
        monetary_value = st.number_input("Monetary Value ($)", min_value=0.0, value=150.0)
        tenure = st.number_input("Customer Tenure (days)", min_value=1, value=180)
        
        submitted = st.form_submit_button("Predict CLV")
        if submitted:
            # This would call the actual prediction API in production
            st.success(f"Predicted CLV: ${1250.50:.2f}")
            st.info("Customer Segment: Champions")

if __name__ == "__main__":
    create_monitoring_dashboard()