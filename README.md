## Customer Churn Prediction System 
A comprehensive machine learning solution for predicting customer churn, built with Python and scikit-learn. This project includes end-to-end implementation from data preprocessing to model deployment.
<img width="656" height="140" alt="image" src="https://github.com/user-attachments/assets/1b1a96d8-fad7-4903-bd50-1c9e24a47da5" />


Project Overview
Customer churn prediction helps businesses identify customers who are likely to stop using their services. This system uses machine learning to predict churn probability, enabling proactive retention strategies and reducing customer attrition.

Features
Complete ML Pipeline: Data preprocessing, feature engineering, model training, and evaluation

Multiple Algorithms: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, SVM

Class Imbalance Handling: SMOTE for balanced training

Comprehensive Evaluation: Multiple metrics and visualizations

Model Interpretation: SHAP values for explainable AI

Deployment Ready: Serialized model and prediction API

Project Structure
text
customer-churn-prediction/
├── churn_prediction.ipynb          # Main Jupyter/Colab notebook
├── requirements.txt                # Dependencies
├── models/
│   └── churn_prediction_model.pkl  # Saved model
├── src/
│   ├── data_preprocessing.py       # Data cleaning and feature engineering
│   ├── model_training.py           # Model training pipeline
│   ├── evaluation.py               # Model evaluation metrics
│   └── prediction.py               # Inference functions
├── data/
│   └── Telco-Customer-Churn.csv    # Sample dataset
└── README.md                       # This file

Installation & Setup
Option 1: Google Colab (Recommended)
Open Google Colab
Upload the notebook file
Run all cells sequentially
The notebook will automatically install required packages

Option 2: Local Setup
bash
# Clone the repository
git clone https://github.com/axestein/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook churn_prediction.ipynb
📦 Dependencies
txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
imbalanced-learn>=0.9.0
shap>=0.40.0
matplotlib>=3.5.0
seaborn>=0.11.0
joblib>=1.1.0
📈 Model Performance
The system evaluates multiple algorithms and selects the best performing model:

Model	Accuracy	Precision	Recall	F1-Score	ROC AUC
XGBoost	0.8321	0.7012	0.5348	0.6065	0.8863
Random Forest	0.7971	0.6435	0.4565	0.5333	0.8431
Gradient Boosting	0.8057	0.6452	0.5174	0.5745	0.8559
Logistic Regression	0.7329	0.5000	0.6435	0.5625	0.7946
SVM	0.7329	0.5000	0.6435	0.5625	0.7912
🔧 Usage
1. Training the Model
python
# The main notebook handles complete training
# Just run all cells in churn_prediction.ipynb
2. Making Predictions
python
from src.prediction import predict_churn

# Example customer data
customer_data = {
    'gender': 'Male',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 12,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 65.5,
    'TotalCharges': 788.5
}

# Get prediction
result = predict_churn(customer_data)
print(result)
3. API Deployment
python
# Example Flask API (app.py)
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('models/churn_prediction_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    probability = model.predict_proba(df)[0][1]
    prediction = int(model.predict(df)[0])
    
    return jsonify({
        'churn_probability': float(probability),
        'churn_prediction': prediction,
        'risk_level': 'High' if probability > 0.5 else 'Low'
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
📊 Data Features
Original Features:
Demographics: gender, SeniorCitizen, Partner, Dependents

Account Information: tenure, Contract, PaperlessBilling, PaymentMethod

Services: PhoneService, MultipleLines, InternetService, OnlineSecurity, etc.

Charges: MonthlyCharges, TotalCharges

Engineered Features:
TenureToMonthlyChargeRatio: Tenure divided by monthly charges

TotalServices: Count of active services

🎯 Key Insights
Top Predictive Features (from SHAP analysis):

Contract type (Month-to-month highest risk)

Tenure length (newer customers higher risk)

Internet service type

Payment method (electronic check highest risk)

High-Risk Customer Profile:

Month-to-month contract

Electronic check payment

High monthly charges

Short tenure

No tech support or online security

📈 Results Visualization
The system generates several visualizations:

Confusion Matrices: For each model

ROC Curves: Comparison of all models

Feature Importance: SHAP summary plots

Churn Distribution: Class balance visualization

🚀 Deployment Strategies
1. Batch Prediction
bash
# Run predictions on entire customer database
python src/batch_prediction.py --input data/customers.csv --output predictions.csv
2. Real-time API
python
# Deploy with Flask/FastAPI for real-time predictions
# Integrate with CRM systems
3. Cloud Deployment
AWS: SageMaker for training, Lambda for inference

GCP: AI Platform, Cloud Functions

Azure: Machine Learning Service

4. Dashboard Integration
python
# Streamlit dashboard example
streamlit run dashboard.py
📋 Monitoring & Maintenance
Model Drift Detection
python
# Monitor model performance over time
from src.monitoring import detect_drift

drift_results = detect_drift(current_data, reference_data)
if drift_results['drift_detected']:
    retrain_model()
Retraining Pipeline
bash
# Automated retraining script
python src/retrain.py --new-data data/new_customers.csv
🔍 Model Interpretation
The system uses SHAP (SHapley Additive exPlanations) to provide:

Feature importance rankings

Individual prediction explanations

Global model behavior insights

🤝 Contributing
Fork the repository

Create a feature branch (git checkout -b feature/AmazingFeature)

Commit changes (git commit -m 'Add AmazingFeature')

Push to branch (git push origin feature/AmazingFeature)

Open a Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
IBM for the Telco Customer Churn dataset

scikit-learn, XGBoost, and SHAP communities
