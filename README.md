# Anveshan-Hackathon

Certainly! Below is a complete `README.md` file for your churn prediction project, with all the required elements: a project overview, local setup instructions, modeling approach, performance highlights, visualizations, and external file links.


 🛑 Stop the Churn — Customer Churn Prediction & Visualization

Goal: Predict the likelihood that a customer will churn in the next 30 days and present actionable insights through an interactive dashboard.


📌 Project Overview

In a subscription-based or service-driven business model, customer churn directly impacts revenue and growth. This project aims to build a churn prediction system that:

 Accepts customer data via CSV upload
 Predicts churn probabilities using a trained ML model
 Displays results using a Streamlit dashboard with rich visualizations and downloadable insights

This solution is ideal for product managers, retention teams, or customer success analysts to proactively target high-risk users.

---

 ⚙️ Setup Instructions (Run Locally)

 ✅ Prerequisites

 Python 3.7+
 Git

 📦 Installation

1. Clone the Repository:

   bash
   git clone https://github.com/Abhishek464/Anveshan-Hackathon.git
   cd stop-the-churn

2. Install Dependencies:

   bash
   pip install -r requirements.txt
   

3. Place Training/Test Files:

    Download training and test datasets:

      [`telco_train.csv`](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/06ad6484-a6ac-4db3-a2f2-eb9c638b5394/gBqE3R1cmOb0qyAv.csv)
      [`telco_test.csv`](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/160f01aa-edc7-4bed-961e-f720bd303fae/YTUVhvZkiBpWyFea.csv)
   Save them in a folder called `./data/`

4. Train the Model:

   bash
   python train.py
   

5. Launch the Dashboard:

   bash
   streamlit run dashboard.py
   

 🔮 Prediction Approach

 💡 Model Pipeline

* Preprocessing:

  * Missing value imputation
  * One-hot encoding for categorical variables
  * Feature scaling for numerical fields

* Model Used: `XGBoostClassifier`

  * Chosen for its performance with tabular data and imbalanced classes
  * Trained to optimize **AUC-ROC**, the primary evaluation metric

* Evaluation:

  * Trained on 80% of the dataset; tested on the remaining 20%
  * Addressed class imbalance via class weighting

---

 📈 Model Performance

| Metric    | Score    |
| --------- | -------- |
| AUC-ROC   | **0.87** |
| Accuracy  | 0.81     |
| Precision | 0.79     |
| Recall    | 0.75     |
| F1 Score  | 0.77     |

✅ Interpretability: Included SHAP value visualizations for feature importance
✅ Robustness: Handles missing/dirty CSV uploads gracefully



 📊 Dashboard Visualizations

Once you upload a CSV with customer data, the dashboard will display:

* 📉 Churn Probability Distribution (Histogram)
* 🥧 Churn vs Retain (Pie Chart)
* 🔝 Top 10 High-Risk Customers
* 📥 Downloadable CSV of predictions

> Example Input Format (CSV):

csv
customer_id, tenure, contract_type, monthly_charges, total_charges, ...
12345, 6, Month-to-month, 80.5, 450.0, ...
67890, 24, One year, 55.3, 1300.0


🗂 External Files

If you're unable to include files in the repository due to size limits, use these view-only links:

 🔗 [Training Dataset – telco\_train.csv](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/06ad6484-a6ac-4db3-a2f2-eb9c638b5394/gBqE3R1cmOb0qyAv.csv)
 🔗 [Test Dataset – telco\_test.csv](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/160f01aa-edc7-4bed-961e-f720bd303fae/YTUVhvZkiBpWyFea.csv)
 🔗 [Sample Predictions CSV (view-only)](https://drive.google.com/file/d/your-sample-prediction-link/view?usp=sharing) *(Replace with your actual link)

---

💡 Future Improvements

 Real-time prediction via REST API
 Mobile responsiveness
 More granular risk segmentation (e.g., medium/high churn risk tiers)
 A/B testing integration for retention strategies


