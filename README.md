Customer Lifetime Value (CLV) Prediction — E-Commerce Analytics Project

This project presents a Computational Statistics framework to calculate and predict the Customer Lifetime Value (CLV) for an e-commerce business.
It uses Python, statistical modeling, cohort analysis, RFM segmentation, and predictive analytics (BG/NBD & Gamma-Gamma models) to identify high-value customers and forecast future revenue.

🚀 Project Objective

The main goal of this project is to:

Calculate CLV using statistical + machine learning methods

Identify top high-value and at-risk customers

Perform RFM Segmentation (Recency–Frequency–Monetary)

Build predictive models for future purchase probability

Help e-commerce businesses allocate marketing budget efficiently

📂 Dataset Description

A real-world e-commerce dataset (transactions-level) including:

Column	Description
CustomerID	Unique customer identifier
InvoiceNo	Unique invoice number
InvoiceDate	Order date
Quantity	Number of items purchased
UnitPrice	Price per item
Country	Customer location
TotalAmount	Computed as Quantity × UnitPrice

Dataset is cleaned and engineered before analysis.

🧹 Data Preprocessing

✔ Missing value handling
✔ Removing cancellations & negative quantities
✔ Converting date columns into datetime
✔ Feature engineering:

Revenue per order

Days since last purchase

Purchase frequency

Average order value (AOV)

✔ Outlier treatment using IQR method

📊 Statistical Analysis Performed
1️⃣ RFM Analysis

Recency (R): Days since last purchase

Frequency (F): Number of orders

Monetary (M): Total amount spent

Customers are segmented into:

🥇 Champions

🥈 Loyal Customers

🟡 Potential Loyalists

🔥 At-Risk Customers

❄️ Hibernating Customers

2️⃣ Cohort Retention Analysis

Monthly cohorts

Retention matrix

Customer repeat-purchase behavior insights

3️⃣ Predictive Modeling for CLV

Using Lifetimes Library:

BG/NBD Model → Predicts future purchasing probability
Gamma-Gamma Model → Predicts expected monetary value

📌 Final CLV = Expected #transactions × Expected profit per transaction

🤖 Machine Learning Model (Optional Extension)

A regression-based ML model is used to compare results with statistical CLV:

Linear Regression

Random Forest Regression

XGBoost Regressor

Performance metrics include:

MAE

RMSE

R² Score

📈 Dashboard & Visualizations

✔ RFM heatmap
✔ Cohort retention matrix
✔ CLV distribution plot
✔ Customer segments pie chart
✔ Probability of future purchases

Tools used: Matplotlib, Seaborn, Plotly

🏗 Project Structure
📁 CLV-Prediction-Framework
 ┣ 📂 data
 ┣ 📂 notebooks
 ┣ 📂 src
 ┣ 📄 requirements.txt
 ┣ 📄 README.md
 ┣ 📄 clv_analysis.py

🛠 Tech Stack
Tool	Purpose
Python	Core programming
Pandas, NumPy	Data preprocessing
Matplotlib, Seaborn, Plotly	Visualizations
Scikit-learn	Machine Learning
Lifetimes Library	Statistical CLV models
Jupyter Notebook	Analysis
📑 Results & Insights
Key Insights:

% of customers contribute to the majority of revenue

Loyal customers show high frequency and high monetary value

At-risk customers identified for targeted remarketing

Predicted CLV helps business plan marketing budget & retention strategies

📎 How to Run the Project
git clone https://github.com/yourusername/CLV-Project.git
cd CLV-Project
pip install -r requirements.txt
python clv_analysis.py

🤝 Future Enhancements

Add Deep Learning sequence models for purchase forecasting

Build a full interactive dashboard using Streamlit

Integrate churn prediction model

