from app import app

# Minimal smoke test using Flask test client

def run_smoke_test():
    app.testing = True
    client = app.test_client()

    form_data = {
        'SeniorCitizen': '1',
        'Partner': '1',
        'Dependents': '0',
        'PhoneService': '1',
        'MultipleLines': '1',
        'PaperlessBilling': '1',
        'OnlineSecurity': '1',
        'OnlineBackup': '1',
        'DeviceProtection': '1',
        'TechSupport': '1',
        'StreamingTV': '1',
        'StreamingMovies': '1',
        'gender': '1',
        'InternetService': '2',  # Fiber Optic
        'Contract': '1',         # One year
        'PaymentMethod': '2',    # Electronic Check
        'MonthlyCharges': '80',
        'Tenure': '12',
    }

    resp = client.post('/predict', data=form_data)
    print('Status:', resp.status_code)
    text = resp.get_data(as_text=True)
    # Print a small snippet for verification
    marker_idx = text.find('Prediction Results')
    print('Contains Prediction Results:', marker_idx != -1)
    print(text[marker_idx:marker_idx+200] if marker_idx != -1 else text[:200])


if __name__ == '__main__':
    run_smoke_test()
