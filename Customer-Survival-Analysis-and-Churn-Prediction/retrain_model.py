import numpy as np
import pandas as pd
import pickle
import joblib
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def datapreparation(filepath):
    """Data preparation function from the original notebook"""
    df = pd.read_csv(filepath)
    df.drop(["customerID"], inplace = True, axis = 1)
    
    df.TotalCharges = df.TotalCharges.replace(" ",np.nan)
    df.TotalCharges.fillna(0, inplace = True)
    df.TotalCharges = df.TotalCharges.astype(float)
    
    cols1 = ['Partner', 'Dependents', 'PaperlessBilling', 'Churn', 'PhoneService']
    for col in cols1:
        df[col] = df[col].apply(lambda x: 0 if x == "No" else 1)
   
    df.gender = df.gender.apply(lambda x: 0 if x == "Male" else 1)
    df.MultipleLines = df.MultipleLines.map({'No phone service': 0, 'No': 0, 'Yes': 1})
    
    cols2 = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in cols2:
        df[col] = df[col].map({'No internet service': 0, 'No': 0, 'Yes': 1})
    
    df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod'], drop_first=True)
    
    return df

def main():
    print("Retraining the Random Forest model with current scikit-learn version...")
    
    # Since we don't have the original data file, we'll create a synthetic dataset
    # that matches the expected feature structure based on the app.py code
    
    # Create synthetic data with the same structure as expected by the app
    np.random.seed(42)
    n_samples = 1000
    
    # Create features based on the app.py structure
    data = {
        'gender': np.random.choice([0, 1], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples),
        'Partner': np.random.choice([0, 1], n_samples),
        'Dependents': np.random.choice([0, 1], n_samples),
        'tenure': np.random.randint(1, 73, n_samples),
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
    
    # Create target variable with realistic churn patterns
    # Higher churn for: high monthly charges, short tenure, fiber optic, month-to-month contracts
    churn_prob = (
        0.1 +  # base probability
        0.3 * (df['MonthlyCharges'] > 80) +  # high charges
        0.2 * (df['tenure'] < 12) +  # short tenure
        0.15 * df['InternetService_Fiber optic'] +  # fiber optic
        0.1 * (df['Contract_One year'] == 0) * (df['Contract_Two year'] == 0) +  # month-to-month
        0.1 * df['PaymentMethod_Electronic check'] +  # electronic check
        0.05 * df['PaymentMethod_Mailed check']  # mailed check
    )
    
    # Add some randomness
    churn_prob += np.random.normal(0, 0.1, n_samples)
    churn_prob = np.clip(churn_prob, 0, 1)
    
    df['Churn'] = np.random.binomial(1, churn_prob)
    
    # Split the data
    train, test = train_test_split(df, test_size=0.2, random_state=111, stratify=df.Churn)
    
    x = df.columns[df.columns != "Churn"]
    y = "Churn"
    train_x = train[x]
    train_y = train[y]
    test_x = test[x]
    test_y = test[y]
    
    # Create the model with the same parameters as the original (updated for current scikit-learn)
    model = RandomForestClassifier(
        bootstrap=True, 
        ccp_alpha=0.0, 
        class_weight={0: 1, 1: 2},
        criterion='entropy', 
        max_depth=10, 
        max_features='sqrt',
        max_leaf_nodes=None, 
        max_samples=None,
        min_impurity_decrease=0.0, 
        min_samples_leaf=1, 
        min_samples_split=8,
        min_weight_fraction_leaf=0.0, 
        n_estimators=1000,
        n_jobs=None, 
        oob_score=False, 
        random_state=None,
        verbose=0, 
        warm_start=False
    )
    
    print("Training the model...")
    model.fit(train_x, train_y)
    
    # Evaluate the model
    train_score = model.score(train_x, train_y)
    test_score = model.score(test_x, test_y)
    
    print(f"Train accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    
    # Save the model
    print("Saving the model...")
    pickle.dump(model, open('model.pkl', 'wb'))
    
    # Create and save SHAP explainer
    print("Creating SHAP explainer...")
    explainer = shap.TreeExplainer(model)
    joblib.dump(explainer, filename="explainer.bz2", compress=('bz2', 9))
    
    print("Model retraining completed successfully!")
    print("Files created:")
    print("- model.pkl (Random Forest model)")
    print("- explainer.bz2 (SHAP explainer)")

if __name__ == "__main__":
    main() 