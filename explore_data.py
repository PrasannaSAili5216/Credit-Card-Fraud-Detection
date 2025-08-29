import pandas as pd

# Load the dataset
df = pd.read_csv('archive/creditcard.csv')

# Display basic information
print('Dataset Info:')
print(df.info())

print('\nDataset Head:')
print(df.head())

print('\nDataset Description:')
print(df.describe())

print('\nClass Distribution:')
print(df['Class'].value_counts())

print('\nMissing Values:')
print(df.isnull().sum())
