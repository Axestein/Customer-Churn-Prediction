import numpy as np
import pickle
import matplotlib
# Use a non-interactive backend for server environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle
import time
from flask import Flask, request, jsonify, render_template
import base64
import io
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# Load models relative to this file's directory
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'model.pkl'
SURV_MODEL_PATH = BASE_DIR / 'survivemodel.pkl'

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(SURV_MODEL_PATH, 'rb') as f:
    survmodel = pickle.load(f)

# Backward-compatibility patch for scikit-learn DecisionTreeClassifier pickles
def _patch_monotonic_constraints(est):
    try:
        # Patch individual DecisionTreeClassifier instances
        if not hasattr(est, 'monotonic_cst'):
            setattr(est, 'monotonic_cst', None)
    except Exception:
        pass

try:
    # If model is an ensemble with base estimators, patch all
    if hasattr(model, 'estimators_'):
        for est in getattr(model, 'estimators_', []):
            _patch_monotonic_constraints(est)
    else:
        # Patch top-level model just in case
        _patch_monotonic_constraints(model)
except Exception:
    # Non-fatal: continue without patch if anything unexpected happens
    pass

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    try:
        # Parse and validate incoming form data (cast strings to numbers)
        gender = int(request.form.get("gender", 0))
        SeniorCitizen = 0
        if 'SeniorCitizen' in request.form:
            SeniorCitizen = 1
        Partner = 0
        if 'Partner' in request.form:
            Partner = 1
        Dependents = 0
        if 'Dependents' in request.form:
            Dependents = 1
        PaperlessBilling = 0
        if 'PaperlessBilling' in request.form:
            PaperlessBilling = 1

        MonthlyCharges = float(request.form.get("MonthlyCharges", 0))
        Tenure = int(float(request.form.get("Tenure", 0)))
        TotalCharges = MonthlyCharges*Tenure

        PhoneService = 0
        if 'PhoneService' in request.form:
            PhoneService = 1

        MultipleLines = 0
        if 'MultipleLines' in request.form and PhoneService == 1:
            MultipleLines = 1

        InternetService = int(request.form.get("InternetService", 0))
        InternetService_Fiberoptic = 0
        InternetService_No = 0
        if InternetService == 0:
            InternetService_No = 1
        elif InternetService == 2:
            InternetService_Fiberoptic = 1

        OnlineSecurity = 0
        if 'OnlineSecurity' in request.form and InternetService_No == 0:
            OnlineSecurity = 1

        OnlineBackup = 0
        if 'OnlineBackup' in request.form and InternetService_No == 0:
            OnlineBackup = 1

        DeviceProtection = 0
        if 'DeviceProtection' in request.form and InternetService_No == 0:
            DeviceProtection = 1

        TechSupport = 0
        if 'TechSupport' in request.form and InternetService_No == 0:
            TechSupport = 1

        StreamingTV = 0
        if 'StreamingTV' in request.form and InternetService_No == 0:
            StreamingTV = 1

        StreamingMovies = 0
        if 'StreamingMovies' in request.form and InternetService_No == 0:
            StreamingMovies = 1

        Contract = int(request.form.get("Contract", 0))
        Contract_Oneyear = 0
        Contract_Twoyear = 0
        if Contract == 1:
            Contract_Oneyear = 1
        elif Contract == 2:
            Contract_Twoyear = 1

        PaymentMethod = int(request.form.get("PaymentMethod", 0))
        PaymentMethod_CreditCard = 0
        PaymentMethod_ElectronicCheck = 0
        PaymentMethod_MailedCheck = 0
        if PaymentMethod == 1:
            PaymentMethod_CreditCard = 1
        elif PaymentMethod == 2:
            PaymentMethod_ElectronicCheck = 1
        elif PaymentMethod == 3:
            PaymentMethod_MailedCheck = 1

        features = [gender, SeniorCitizen, Partner, Dependents, Tenure, PhoneService, MultipleLines, OnlineSecurity, OnlineBackup,
           DeviceProtection, TechSupport, StreamingTV, StreamingMovies, PaperlessBilling, MonthlyCharges, TotalCharges,
           InternetService_Fiberoptic, InternetService_No, Contract_Oneyear,Contract_Twoyear,
           PaymentMethod_CreditCard, PaymentMethod_ElectronicCheck, PaymentMethod_MailedCheck]

        columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
           'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies','PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
           'InternetService_Fiber optic', 'InternetService_No', 'Contract_One year', 'Contract_Two year',
           'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

        # Create a DataFrame with proper feature names
        final_features_df = pd.DataFrame([features], columns=columns)

        # Predict probability with fallbacks
        try:
            if hasattr(model, "predict_proba"):
                prediction = model.predict_proba(final_features_df)
                # Normalize to valid probability distribution if needed
                row = np.array(prediction[0], dtype=float)
                # Try to locate the positive class index (class label == 1)
                pos_idx = 1
                try:
                    if hasattr(model, 'classes_'):
                        classes = np.array(model.classes_)
                        if 1 in classes:
                            pos_idx = int(np.where(classes == 1)[0][0])
                        elif len(classes) == 2:
                            # If binary but label 1 not present, assume the higher label is positive
                            pos_idx = int(np.argmax(classes))
                except Exception:
                    pass

                s = row.sum()
                if not np.isfinite(s) or s <= 0 or s < 0.99 or s > 1.01:
                    # Use softmax to convert logits/scores to probabilities
                    m = np.max(row)
                    exps = np.exp(row - m)
                    row = exps / np.sum(exps)
                else:
                    row = row / s
                output = float(np.clip(row[pos_idx], 0.0, 1.0))
            elif hasattr(model, "decision_function"):
                # Map decision function to probability via logistic transform as a fallback
                df_val = float(model.decision_function(final_features_df)[0])
                output = 1.0 / (1.0 + np.exp(-df_val))
            else:
                # Binary prediction fallback
                pred = int(model.predict(final_features_df)[0])
                output = float(pred)
            # Ensure probability is within [0, 1]
            output = float(np.clip(output, 0.0, 1.0))
        except Exception as pred_err:
            raise RuntimeError(f"Prediction step failed: {pred_err}")
        # Build a feature importance plot (avoid SHAP explainer incompatibilities)
        shap_img = io.BytesIO()
        plt.figure(figsize=(10, 6))
        feature_names = columns
        try:
            if hasattr(model, "feature_importances_"):
                feature_importance = np.array(getattr(model, "feature_importances_"))
                sorted_idx = np.argsort(feature_importance)
                pos = np.arange(len(sorted_idx)) + 0.5
                plt.barh(pos, feature_importance[sorted_idx])
                plt.yticks(pos, np.array(feature_names)[sorted_idx])
                plt.xlabel('Feature Importance')
                plt.title('Tree-based Feature Importance')
            elif hasattr(model, "coef_"):
                coef = np.ravel(np.abs(getattr(model, "coef_")))
                sorted_idx = np.argsort(coef)
                pos = np.arange(len(sorted_idx)) + 0.5
                plt.barh(pos, coef[sorted_idx])
                plt.yticks(pos, np.array(feature_names)[sorted_idx])
                plt.xlabel('Coefficient Magnitude (abs)')
                plt.title('Linear Model Coefficients')
            else:
                plt.text(0.5, 0.5, 'Feature importance not available for this model.',
                         ha='center', va='center', fontsize=12)
                plt.axis('off')
        except Exception as imp_err:
            plt.clf()
            plt.figure(figsize=(6, 3))
            plt.text(0.5, 0.5, f'Importance plot failed: {imp_err}', ha='center', va='center')
            plt.axis('off')
        finally:
            plt.tight_layout()
            plt.savefig(shap_img, bbox_inches="tight", format='png')
            plt.close()
        shap_img.seek(0)
        shap_url = base64.b64encode(shap_img.getvalue()).decode()

        # Hazard and Survival Analysis
        surv_columns = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
            'MonthlyCharges', 'TotalCharges', 'InternetService_Fiber optic', 'InternetService_No', 'Contract_One year',
            'Contract_Two year', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
            'PaymentMethod_Mailed check'
        ]
        surv_values = [
            gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines, OnlineSecurity, OnlineBackup,
            DeviceProtection, TechSupport, StreamingTV, StreamingMovies, PaperlessBilling, MonthlyCharges, TotalCharges,
            InternetService_Fiberoptic, InternetService_No, Contract_Oneyear, Contract_Twoyear,
            PaymentMethod_CreditCard, PaymentMethod_ElectronicCheck, PaymentMethod_MailedCheck
        ]
        surv_input_df = pd.DataFrame([surv_values], columns=surv_columns)

        hazard_img = io.BytesIO()
        fig, ax = plt.subplots()
        try:
            survmodel.predict_cumulative_hazard(surv_input_df).plot(ax=ax, color='red')
        except Exception as haz_err:
            raise RuntimeError(f"Cumulative hazard plot failed: {haz_err}")
        plt.axvline(x=Tenure, color='blue', linestyle='--')
        plt.legend(labels=['Hazard', 'Current Position'])
        ax.set_xlabel('Tenure', size=10)
        ax.set_ylabel('Cumulative Hazard', size=10)
        ax.set_title('Cumulative Hazard Over Time')
        plt.savefig(hazard_img, format='png')
        hazard_img.seek(0)
        hazard_url = base64.b64encode(hazard_img.getvalue()).decode()

        surv_img = io.BytesIO()
        fig, ax = plt.subplots()
        try:
            survmodel.predict_survival_function(surv_input_df).plot(ax=ax, color='red')
        except Exception as surv_err:
            raise RuntimeError(f"Survival function plot failed: {surv_err}")
        plt.axvline(x=Tenure, color='blue', linestyle='--')
        plt.legend(labels=['Survival Function', 'Current Position'])
        ax.set_xlabel('Tenure', size=10)
        ax.set_ylabel('Survival Probability', size=10)
        ax.set_title('Survival Probability Over Time')
        plt.savefig(surv_img, format='png')
        surv_img.seek(0)
        surv_url = base64.b64encode(surv_img.getvalue()).decode()

        # Compute expected lifetime (months) where survival probability > 0.1
        try:
            life_df = survmodel.predict_survival_function(surv_input_df)
        except Exception as life_err:
            raise RuntimeError(f"Survival function computation failed: {life_err}")
        # life_df: index=time, columns per sample (one column for single row)
        life_df = life_df.reset_index()
        life_df.columns = ['Tenure'] + [str(c) for c in life_df.columns[1:]]
        # Use the first sample's column
        prob_col = life_df.columns[1]
        life_df.rename(columns={prob_col: 'Probability'}, inplace=True)
        max_life = life_df.Tenure[life_df.Probability > 0.1].max()
        if pd.isna(max_life):
            max_life = float(life_df['Tenure'].max())

        CLTV = max_life * MonthlyCharges

        # Gauge plot
        def degree_range(n):
            start = np.linspace(0,180,n+1, endpoint=True)[0:-1]
            end = np.linspace(0,180,n+1, endpoint=True)[1::]
            mid_points = start + ((end-start)/2.)
            return np.c_[start, end], mid_points

        def rot_text(ang):
            rotation = np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))
            return rotation

        def gauge(labels=['LOW','MEDIUM','HIGH','EXTREME'], \
                  colors=['#007A00','#0063BF','#FFCC00','#ED1C24'], Probability=1, fname=False):

            N = len(labels)
            colors = colors[::-1]


            """
            begins the plotting
            """

            gauge_img = io.BytesIO()
            fig, ax = plt.subplots()

            ang_range, mid_points = degree_range(4)

            labels = labels[::-1]

            """
            plots the sectors and the arcs
            """
            patches = []
            for ang, c in zip(ang_range, colors):
                # sectors
                patches.append(Wedge((0.,0.), .4, *ang, facecolor='w', lw=2))
                # arcs
                patches.append(Wedge((0.,0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=0.5))

            [ax.add_patch(p) for p in patches]


            """
            set the labels (e.g. 'LOW','MEDIUM',...)
            """

            for mid, lab in zip(mid_points, labels):

                ax.text(0.35 * np.cos(np.radians(mid)), 0.35 * np.sin(np.radians(mid)), lab, \
                    horizontalalignment='center', verticalalignment='center', fontsize=14, \
                    fontweight='bold', rotation = rot_text(mid))

            """
            set the bottom banner and the title
            """
            r = Rectangle((-0.4,-0.1),0.8,0.1, facecolor='w', lw=2)
            ax.add_patch(r)

            ax.text(0, -0.05, 'Churn Probability ' + np.round(Probability,2).astype(str), horizontalalignment='center', \
                 verticalalignment='center', fontsize=22, fontweight='bold')

            """
            plots the arrow now
            """

            pos = (1-Probability)*180
            ax.arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 0.225 * np.sin(np.radians(pos)), \
                         width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')

            ax.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
            ax.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))

            """
            removes frame and ticks, and makes axis equal and tight
            """

            ax.set_frame_on(False)
            ax.axes.set_xticks([])
            ax.axes.set_yticks([])
            ax.axis('equal')
            plt.tight_layout()

            plt.savefig(gauge_img, format = 'png')
            gauge_img.seek(0)
            url = base64.b64encode(gauge_img.getvalue()).decode()
            return url

        gauge_url = gauge(Probability = output)

        t = time.time()
        
        # Format the prediction text more nicely
        churn_prob_percent = round(output * 100, 1)
        cltv_formatted = "${:,.2f}".format(CLTV)
        
        if output < 0.3:
            risk_level = "Low Risk"
            risk_color = "#28a745"
        elif output < 0.6:
            risk_level = "Medium Risk"
            risk_color = "#ffc107"
        else:
            risk_level = "High Risk"
            risk_color = "#dc3545"
        
        prediction_text = f"Churn Probability: {churn_prob_percent}% ({risk_level}) | Expected Lifetime Value: {cltv_formatted}"
        
        return render_template('index.html', 
                             prediction_text=prediction_text,
                             churn_probability=churn_prob_percent,
                             risk_level=risk_level,
                             cltv=cltv_formatted,
                             url_1=gauge_url, 
                             url_2=shap_url, 
                             url_3=hazard_url, 
                             url_4=surv_url)
    except Exception as e:
        # Return error page or message
        error_message = f"An error occurred during prediction: {str(e)}"
        return render_template('index.html', 
                             prediction_text=error_message,
                             churn_probability=0,
                             risk_level="Error",
                             cltv="$0.00",
                             url_1="", 
                             url_2="", 
                             url_3="", 
                             url_4="")


if __name__ == "__main__":
    app.run(debug=True)
