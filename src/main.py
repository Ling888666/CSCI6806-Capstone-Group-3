import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score

# Assuming the target variable 'Defective' is binary and present in the dataset.
# Assuming 'Defective' is 1 if the product is defective and 0 otherwise.
# This is a mock column based on paper description.
# In the real case, this should be informed by the actual process of labeling defective products.

# Load data
data = pd.read_csv('./data/SCMS_Delivery_History_Dataset.csv')

# Feature extraction
# Select the columns which were mentioned to be used - assuming that similar columns will be used to construct features.
features = [
    'Line Item Quantity',
    'Line Item Value',
    'Weight (Kilograms)',
    'Freight Cost (USD)',
    'Unit of Measure (Per Pack)',
    'Pack Price',
    'Line Item Insurance (USD)'
]
target = 'Defective' # We need to define how the defective label is assigned based on the dataset


#clean the data cells for the non-numeric value.
def laundry_func(x):
    import math
    try:
        # return -1 if the value is not a float or int.
        return -1 if (math.isnan(float(str(x)))) else x
    except:
        # return -1 if convertion failed
        return -1
data[features] = data[features].applymap(laundry_func)

print(f"Input Data:\n{data[features].fillna(0)}")

# Generating the 'Defective' column for simulation purposes – this could be a complex process based on domain knowledge
threshold = (data['Line Item Value']/data['Line Item Quantity']).mean()*0.3
data[target] = ((data['Line Item Value']/data['Line Item Quantity']) < threshold).astype(int)

print(f"Target Data:\n{data[target].fillna(0)}")

print("Processing Start...")
# Data preprocessing - scaling and normalization
scaler = StandardScaler()

X = scaler.fit_transform(data[features])
y = data[target]

# Splitting the dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing models
linear_reg = LinearRegression()
logistic_reg = LogisticRegression()

# Training the Linear Regression Model
linear_reg.fit(X_train, y_train)
y_pred_linear = linear_reg.predict(X_test)
y_pred_linear = [1 if x > 0.5 else 0 for x in y_pred_linear]  # Thresholding to get binary classification

# Training the Logistic Regression Model
logistic_reg.fit(X_train, y_train)
y_pred_logistic = logistic_reg.predict(X_test)

# Evaluating the models
linear_accuracy = accuracy_score(y_test, y_pred_linear)
linear_precision = precision_score(y_test, y_pred_linear)

logistic_accuracy = accuracy_score(y_test, y_pred_logistic)
logistic_precision = precision_score(y_test, y_pred_logistic)

print("Outputing Results...")

print(f"Accuracy of Linear Regression: {linear_accuracy:.4f}")
print(f"Precision of Linear Regression: {linear_precision:.4f}")
print(f"Accuracy of Logistic Regression: {logistic_accuracy:.4f}")
print(f"Precision of Logistic Regression: {logistic_precision:.4f}")

# Note: In a real replication, you may need to perform cross-validation or use a validation set to tune hyperparameters.