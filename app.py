
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request, jsonify
import numpy as np
import json
import shap

app = Flask(__name__)

# Load the model
model = joblib.load('credit_card_fraud_model.pkl')

# --- Data for pre-filling the form ---
# Read a small chunk of the data to find sample rows
df = pd.read_csv('archive/creditcard.csv', nrows=100000)
fraudulent_sample = df[df['Class'] == 1].iloc[0].to_dict()
non_fraudulent_sample = df[df['Class'] == 0].iloc[0].to_dict()

# Convert numpy types to standard Python types for JSON serialization
for k, v in fraudulent_sample.items():
    if hasattr(v, 'item'):
        fraudulent_sample[k] = v.item()
for k, v in non_fraudulent_sample.items():
    if hasattr(v, 'item'):
        non_fraudulent_sample[k] = v.item()
# --- End of data loading ---

# Load the full dataset for SHAP background
full_df = pd.read_csv('archive/creditcard.csv')
features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
# Use a sample of non-fraudulent data as background for SHAP
# This is important for LinearExplainer
background_data = full_df[full_df['Class'] == 0].sample(100, random_state=42)
background_data = background_data[features].astype(float)


@app.route('/')
def home():
    return render_template('index.html', 
                           fraud_sample=json.dumps(fraudulent_sample), 
                           non_fraud_sample=json.dumps(non_fraudulent_sample))

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()
    
    # Keep a copy of the original form data for rendering the result
    original_form_data = form_data.copy()

    data = pd.DataFrame([form_data])

    # --- Preprocessing ---
    # The V features are already scaled, only scale Time and Amount
    scaler = StandardScaler()
    data['Time'] = scaler.fit_transform(data[['Time']].astype(float))
    data['Amount'] = scaler.fit_transform(data[['Amount']].astype(float))
    # --- End of Preprocessing ---

    features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    data_for_prediction = data[features].astype(float)

    prediction = model.predict(data_for_prediction)[0]
    
    card_number = original_form_data.get('CardNumber', '')
    if len(card_number) > 4:
        card_number_display = '&bull;' * (len(card_number) - 4) + card_number[-4:]
    else:
        card_number_display = card_number

    # Calculate SHAP values if prediction is fraudulent
    top_contributing_features = []
    if prediction == 1:
        # Use LinearExplainer for LogisticRegression for efficiency
        explainer = shap.LinearExplainer(model, background_data)
        shap_values = explainer.shap_values(data_for_prediction)

        # shap_values will be a list of arrays for multi-output models,
        # for binary classification, it's usually one array for class 1
        # or two arrays (one for each class). Take the second one (for class 1).
        if isinstance(shap_values, list):
            shap_values = shap_values[1] # Assuming index 1 is for the positive class

        # Get absolute SHAP values to find most impactful features
        abs_shap_values = np.abs(shap_values[0]) # shap_values[0] because data_for_prediction is a single row

        # Get feature names
        feature_names = data_for_prediction.columns

        # Create a dictionary of feature importance
        feature_importance = dict(zip(feature_names, abs_shap_values))

        # Sort features by importance and get top N (e.g., top 5)
        sorted_features = sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)
        top_contributing_features = sorted_features[:5] # Get top 5 features

    # Pass the original form data back to the template to re-populate the form
    return render_template('index.html', 
                           prediction=prediction, 
                           card_number=card_number_display,
                           form_data=original_form_data,
                           fraud_sample=json.dumps(fraudulent_sample), 
                           non_fraud_sample=json.dumps(non_fraudulent_sample),
                                                          top_contributing_features=top_contributing_features)

@app.route('/transactions')
def transactions():
    df = pd.read_csv('archive/creditcard.csv')
    fraudulent_transactions = df[df['Class'] == 1]
    transactions = fraudulent_transactions.to_dict(orient='records')
    columns = fraudulent_transactions.columns.tolist()
    return render_template('transactions.html', transactions=transactions, columns=columns)

if __name__ == '__main__':
    app.run(debug=True)
