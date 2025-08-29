<img width="981" height="858" alt="Home Page" src="https://github.com/user-attachments/assets/483cf8ab-69c2-4b64-b3c3-0c2f29e22e19" />


# Credit Card Fraud Detection

This project is a machine learning application that detects fraudulent credit card transactions. It includes a Flask web application for real-time predictions and scripts for training and evaluating the model.

## Project Structure

```
├── app.py                  # Main Flask application
├── train_model.py          # Script to train the fraud detection model
├── predict.py              # Script for making batch predictions
├── requirements.txt        # Python dependencies
├── credit_card_fraud_model.pkl # Trained model file
├── new_data.csv            # Sample data for batch prediction
├── archive/
│   └── creditcard.csv      # Dataset (from Kaggle)
├── templates/
│   └── index.html          # HTML template for the web app
├── precision_recall_curve.png # Image of the precision-recall curve
└── confusion_matrix.png    # Image of the confusion matrix
```

## Getting Started

### Prerequisites

- Python 3.x
- pip

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # source venv/bin/activate  # On macOS/Linux
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset:**
   - Download the "Credit Card Fraud Detection" dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).
   - Place the `creditcard.csv` file inside the `archive` directory.

## Usage

### Running the Web Application

To start the Flask server for real-time predictions:

```bash
python app.py
```

Open your web browser and go to `http://127.0.0.1:5000` to access the application.

### Making Batch Predictions

To make predictions on a new dataset, use the `predict.py` script. The sample data is in `new_data.csv`.

```bash
python predict.py
```

The script will output the predictions to the console.

## Model Training and Evaluation

To train the model from scratch, run the `train_model.py` script:

```bash
python train_model.py
```

This script will:
- Load the dataset.
- Preprocess the data (scaling, handling class imbalance with SMOTE).
- Train a Logistic Regression model.
- Evaluate the model and save the following files:
  - `credit_card_fraud_model.pkl`: The trained model.
  - `precision_recall_curve.png`: A plot of the precision-recall curve.
  - `confusion_matrix.png`: A plot of the confusion matrix.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
