import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

file_path = r'/Users/alan/Desktop/FDU/Capstone/SCMS_Delivery_History_Dataset.csv'
data = pd.read_csv(file_path)

print(data.head())

threshold = data['Line Item Value'].mean()
data['Target'] = (data['Line Item Value'] > threshold).astype(int)

X = data[['Line Item Quantity']].values
y = data['Target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"accuracy: {accuracy:.2f}")
print(f"intercept (B0): {model.intercept_[0]}")
print(f"coefficient (B1): {model.coef_[0][0]}")
