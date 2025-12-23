import numpy as np
import pandas as pd
import pickle
from lifelines import CoxPHFitter
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Retraining the Cox Proportional Hazard survival model...")
    
    # Create synthetic data for survival analysis
    np.random.seed(42)
    n_samples = 1000
    
    # Create features for survival analysis (same structure as expected by app.py)
    data = {
        'gender': np.random.choice([0, 1], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples),
        'Partner': np.random.choice([0, 1], n_samples),
        'Dependents': np.random.choice([0, 1], n_samples),
        'PhoneService': np.random.choice([0, 1], n_samples),
        'MultipleLines': np.random.choice([0, 1], n_samples),
        'OnlineSecurity': np.random.choice([0, 1], n_samples),
        'OnlineBackup': np.random.choice([0, 1], n_samples),
        'DeviceProtection': np.random.choice([0, 1], n_samples),
        'TechSupport': np.random.choice([0, 1], n_samples),
        'StreamingTV': np.random.choice([0, 1], n_samples),
        'StreamingMovies': np.random.choice([0, 1], n_samples),
        'PaperlessBilling': np.random.choice([0, 1], n_samples),
        'MonthlyCharges': np.random.uniform(20, 120, n_samples),
        'TotalCharges': np.random.uniform(20, 8000, n_samples),
        'InternetService_Fiber optic': np.random.choice([0, 1], n_samples),
        'InternetService_No': np.random.choice([0, 1], n_samples),
        'Contract_One year': np.random.choice([0, 1], n_samples),
        'Contract_Two year': np.random.choice([0, 1], n_samples),
        'PaymentMethod_Credit card (automatic)': np.random.choice([0, 1], n_samples),
        'PaymentMethod_Electronic check': np.random.choice([0, 1], n_samples),
        'PaymentMethod_Mailed check': np.random.choice([0, 1], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create survival times and events
    # Higher risk for: high monthly charges, fiber optic, month-to-month contracts
    risk_score = (
        1.0 +  # base risk
        0.5 * (df['MonthlyCharges'] > 80) +  # high charges
        0.3 * df['InternetService_Fiber optic'] +  # fiber optic
        0.2 * (df['Contract_One year'] == 0) * (df['Contract_Two year'] == 0) +  # month-to-month
        0.1 * df['PaymentMethod_Electronic check'] +  # electronic check
        0.05 * df['PaymentMethod_Mailed check']  # mailed check
    )
    
    # Generate survival times (in months)
    base_time = np.random.exponential(24, n_samples)  # mean 24 months
    survival_time = base_time / risk_score
    survival_time = np.clip(survival_time, 1, 72)  # between 1 and 72 months
    
    # Create events (1 = churned, 0 = censored)
    # Higher probability of churn for shorter survival times
    churn_prob = 1 - (survival_time / 72)
    events = np.random.binomial(1, churn_prob)
    
    # Add survival data to dataframe
    df['duration'] = survival_time
    df['event'] = events
    
    print(f"Dataset created with {n_samples} samples")
    print(f"Event rate: {events.mean():.3f}")
    print(f"Mean survival time: {survival_time.mean():.1f} months")
    
    # Fit Cox Proportional Hazard model
    print("Fitting Cox Proportional Hazard model...")
    cph = CoxPHFitter()
    cph.fit(df, duration_col='duration', event_col='event')
    
    # Print model summary
    print("\nModel Summary:")
    print(cph.print_summary())
    
    # Save the model
    print("\nSaving the survival model...")
    pickle.dump(cph, open('survivemodel.pkl', 'wb'))
    
    print("Survival model retraining completed successfully!")
    print("File created: survivemodel.pkl")

if __name__ == "__main__":
    main() 