import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score

import json
cfg = None

config_path = './data/config.json'
raw_csv_path = './data/SCMS_Delivery_History_Dataset.csv'
csv_path = './data/SCMS_Delivery_History_Dataset.out.csv'

with open(config_path) as json_data:
    cfg = json.load(json_data)
    print(cfg)

if cfg is not None and cfg['do_laundry']:
    from data_utils import SqlLiteDataFrame
    # My data laundry library that makes the whole dateset usable (All 33 Original Columns are used)
    # See Data_Legends.pdf
    sqlLiteDf = SqlLiteDataFrame(raw_csv_path)
    sqlLiteDf.generate_target(used_ratio=0.3,thresh=0.3)
    sqlLiteDf.to_csv()


def new_stuff(csv_path):
    # read in dataset
    data = pd.read_csv(csv_path)

    # divide input and target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # divide dataset（80% train，20% test）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # init random forest and XGBoost classifier
    rf = RandomForestClassifier(random_state=42)
    xgb_clf = xgb.XGBClassifier(objective='binary:logistic', random_state=42)

    # define hyper grid
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
    }

    xgb_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
    }

    # create RandomizedSearchCV object
    rf_random_search = RandomizedSearchCV(rf, rf_param_grid, n_iter=10, cv=3, 
                                        scoring='accuracy', random_state=42, n_jobs=-1)
    xgb_random_search = RandomizedSearchCV(xgb_clf, xgb_param_grid, n_iter=10, cv=3, 
                                        scoring='accuracy', random_state=42, n_jobs=-1)

    # do hyper args
    print("Tuning Random Forest...")
    rf_random_search.fit(X_train, y_train)
    print("Tuning XGBoost...")
    xgb_random_search.fit(X_train, y_train)

    # find best
    rf_best = rf_random_search.best_estimator_
    xgb_best = xgb_random_search.best_estimator_

    # vote classifier
    voting_clf = VotingClassifier(estimators=[('rf', rf_best), ('xgb', xgb_best)], voting='soft')

    # train voter
    voting_clf.fit(X_train, y_train)

    # predict
    y_pred = voting_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    print(f'Accuracy with ensemble model: {accuracy}')
    print(f'Precision with ensemble model: {precision}')



def old_stuff(csv_path):
    # Assuming the target variable 'Defective' is binary and present in the dataset.
    # Assuming 'Defective' is 1 if the product is defective and 0 otherwise.
    # This is a mock column based on paper description.
    # In the real case, this should be informed by the actual process of labeling defective products.

    # Load data
    data = pd.read_csv(csv_path)

    # # Feature extraction
    # # Select the columns which were mentioned to be used - assuming that similar columns will be used to construct features.
    # features = [
    #     'Line Item Quantity',
    #     'Line Item Value',
    #     'Weight (Kilograms)',
    #     'Freight Cost (USD)',
    #     'Unit of Measure (Per Pack)',
    #     'Pack Price',
    #     'Line Item Insurance (USD)'
    # ]
    # target = 'Defective' # We need to define how the defective label is assigned based on the dataset

    # data.columns.to_list[:, :-1]
    column_names=data.columns.to_list()

    features = column_names[:-1]
    target = column_names[-1]


    def laundry_func(x):
        import math
        try:
            return -1 if (math.isnan(float(str(x)))) else x
        except:
            return -1
    data[features] = data[features].map(laundry_func)

    print(f"Input Data:\n{data[features].fillna(0).head()}")

    # # Generating the 'Defective' column for simulation purposes – this could be a complex process based on domain knowledge
    # threshold = (data['Line Item Value']/data['Line Item Quantity']).mean()*0.3
    # data[target] = ((data['Line Item Value']/data['Line Item Quantity']) < threshold).astype(int)

    print(f"Target Data:\n{data[target].fillna(0).head()}")

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

if cfg is not None and cfg['legacy_only']:
    old_stuff(csv_path)
else:
    print("--------------------old version--------------------")
    old_stuff(csv_path)
    print("--------------------new version--------------------")
    new_stuff(csv_path)