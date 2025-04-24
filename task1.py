import pandas as pd
import numpy as np

# Create the dataset
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Ethan', 'Fiona', 'George', 'Hannah', 'Ian', 'Jane'],
    'Age': [25, np.nan, 28, 35, 40, 30, 22, 29, 33, np.nan],
    'Salary': [50000, 62000, np.nan, 70000, 72000, 68000, 48000, 69000, 71000, 73000],
    'Department': ['HR', 'IT', 'Marketing', 'HR', 'IT', 'Marketing', 'HR', 'IT', 'Marketing', 'HR'],
    'Gender': ['Female', 'Male', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'Experience': [1.5, 2.0, 1.0, np.nan, 5.0, 3.5, 0.5, 3.0, 4.0, 4.5]
}


df = pd.DataFrame(data)

df.to_csv('sample_dataset.csv', index=False)

print(" Sample dataset created and saved as 'sample_dataset.csv'")

df = pd.read_csv('sample_dataset.csv')

print("\nMissing Values:\n", df.isnull().sum())

print("\nData Types:\n", df.dtypes)

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Experience'] = df['Experience'].fillna(df['Experience'].median())
df['Salary'] = df['Salary'].fillna(df['Salary'].mean())
 
df_encoded = pd.get_dummies(df, columns=['Department', 'Gender'], drop_first=True)
print(df_encoded.head())

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_encoded[['Age', 'Salary', 'Experience']] = scaler.fit_transform(df_encoded[['Age', 'Salary', 'Experience']])
print(df_encoded.head())

import seaborn as sns
import matplotlib.pyplot as plt


plt.figure(figsize=(12, 4))
for i, col in enumerate(['Age', 'Salary', 'Experience']):
    plt.subplot(1, 3, i+1)
    sns.boxplot(y=df_encoded[col])
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()
Q1 = df_encoded['Experience'].quantile(0.25)
Q3 = df_encoded['Experience'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter
df_cleaned = df_encoded[(df_encoded['Experience'] >= lower_bound) & (df_encoded['Experience'] <= upper_bound)]

print("\nData shape before outlier removal:", df_encoded.shape)
print("Data shape after outlier removal:", df_cleaned.shape)
print(df)





