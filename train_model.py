import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the dataset
df = pd.read_csv('archive/creditcard.csv')

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Feature Scaling for 'Time' and 'Amount'
scaler = StandardScaler()
X['Time'] = scaler.fit_transform(X[['Time']])
X['Amount'] = scaler.fit_transform(X[['Amount']])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print('\nClass Distribution after SMOTE:')
print(pd.Series(y_train_resampled).value_counts())

# Model Training (Logistic Regression)
model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced') # Using class_weight for initial handling
model.fit(X_train_resampled, y_train_resampled)

# Model Evaluation
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print('\nConfusion Matrix:')
print(confusion_matrix(y_test, y_pred))

print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# Precision-Recall Curve and AUC-PR
average_precision = average_precision_score(y_test, y_pred_proba)
print(f'\nAverage Precision Score (AP): {average_precision:.4f}')

precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
auc_pr = auc(recall, precision)
print(f'Area Under the Precision-Recall Curve (AUC-PR): {auc_pr:.4f}')

# Plotting Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'Precision-Recall curve (AP = {average_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(True)
plt.savefig(os.path.join(os.getcwd(), 'precision_recall_curve.png'))
print('Precision-Recall Curve saved as precision_recall_curve.png')

# Plotting Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(os.getcwd(), 'confusion_matrix.png'))
print('Confusion Matrix saved as confusion_matrix.png')

# Save the trained model
import joblib
joblib.dump(model, 'credit_card_fraud_model.pkl')
print('Trained model saved as credit_card_fraud_model.pkl')
