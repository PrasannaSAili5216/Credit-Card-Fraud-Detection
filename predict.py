import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

def predict_fraud(data_path):
    """Loads the trained model and makes predictions on new data."""
    # Load the trained model
    model = joblib.load('credit_card_fraud_model.pkl')

    # Load new data
    new_data = pd.read_csv(data_path)

    # Preprocess the new data (similar to training)
    scaler = StandardScaler()
    new_data['Time'] = scaler.fit_transform(new_data[['Time']])
    new_data['Amount'] = scaler.fit_transform(new_data[['Amount']])

    # Make predictions
    predictions = model.predict(new_data)
    prediction_probabilities = model.predict_proba(new_data)[:, 1]

    # Add predictions to the dataframe
    new_data['Predicted_Class'] = predictions
    new_data['Prediction_Probability'] = prediction_probabilities

    return new_data

if __name__ == '__main__':
    # Path to the new data
    new_data_path = 'new_data.csv'

    # Get predictions
    predictions_df = predict_fraud(new_data_path)

    # Display the predictions
    print('Predictions on New Data:')
    print(predictions_df)
