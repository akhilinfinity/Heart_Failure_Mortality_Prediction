❤️ Heart Failure Mortality Prediction
📌 Overview

This project focuses on predicting the likelihood of mortality in patients with heart failure using clinical records. By analyzing medical attributes such as age, blood pressure, serum creatinine, ejection fraction, and more, we aim to identify key factors that influence survival and build predictive models to assist healthcare providers in decision-making.

🎯 Problem Statement

Heart failure is a critical condition where the heart cannot pump enough blood to meet the body’s needs. Timely prediction of patient mortality risk can:

Help doctors prioritize high-risk patients.

Support better clinical decisions.

Potentially save lives by guiding early interventions.

🛠️ Project Workflow

Import Dataset – Load the Heart Failure Clinical Records dataset.

Data Inspection & Cleaning

Check for missing values & datatypes

Handle outliers if necessary

Exploratory Data Analysis (EDA)

Univariate analysis: distribution of features like age, ejection fraction, creatinine levels

Bivariate analysis: relationship of features with mortality

Correlation heatmap for identifying strong predictors

Feature Engineering

Scaling/normalization of numerical variables

Encoding categorical variables

Model Building

Train different machine learning models (Logistic Regression, Decision Trees, Random Forest, etc.)

Evaluate performance with accuracy, precision, recall, F1-score, and ROC-AUC

Insights & Recommendations

📂 Dataset

Source: Heart Failure Clinical Records Dataset (UCI)

Size: 299 patients, 13 clinical features

Target Variable: DEATH_EVENT

0 = Patient survived

1 = Patient died

🛠️ Tech Stack

Language: Python

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

Environment: Jupyter Notebook

📊 Key Insights (Expected)

Patients with low ejection fraction and high serum creatinine have higher mortality risk.

Age is an important factor, with older patients more likely to face adverse outcomes.

Machine learning models can effectively classify patients into high-risk and low-risk groups.

🚀 How to Run

Clone this repo:

git clone https://github.com/akhilinfinity/Heart-Failure-Mortality-Prediction.git
cd Heart-Failure-Mortality-Prediction


Install dependencies:

pip install -r requirements.txt


Open Jupyter Notebook and run heart_failure_prediction.ipynb.

📌 Future Work

Apply advanced ML models like XGBoost, LightGBM, and Neural Networks.

Deploy model using Streamlit/Flask for clinical usability.

Build a dashboard for interactive visualization.
