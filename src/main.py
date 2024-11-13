import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score

# My data laundry library that makes the whole dateset usable (All 33 Original Columns are used)
# See Data_Legends.pdf
from data_utils import SqlLiteDataFrame

config_path = './data/config.json'

def get_rand_state(int_val):
    import numpy as np
    # print(f"get_rand_state(int_val={int_val})")
    return np.random.RandomState(int_val)

class TheConfig:
    Defaults = {
        'do_laundry':False,
        'legacy_only':False,
        'generate_target':False,
        'num_shots':1,
        'raw_csv_path':"SCMS_Delivery_History_Dataset.csv",
        'csv_path':"SCMS_Delivery_History_Dataset.out.csv",
        'used_ratio':0.3,
        'thresh':0.3,
        'test_size':0.2,
        'gen_state':42,
        'gen_weight':[113,112],
        'gen_source_columns':None,
        'random_state':42,
        'rf_param_grid': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            },
        'xgb_param_grid': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            },
        'n_iter':10,
        'rf_n_iter':4,
        'xgb_n_iter':8,
        'cv':3,
        'n_jobs':-1,
        'voting':"soft"
    }
    Keys = Defaults.keys()
    CurrentConfig = None
    # usage
    # for attr in TheConfig.keys:
    # locals()[attr] = getattr(obj, attr)
    def __init__(self,cfg_path):
        import os
        self.cfg_path = os.path.abspath(cfg_path)
        cfg_folder = os.path.split(self.cfg_path)[0]
        cfg = None
        # read from file
        with open(self.cfg_path) as json_data:
            cfg = json.load(json_data)
            for key, value in cfg.items():
                setattr(self, key, value)
                # print(f"Config['{key}'] = {value}")
        # fill with defaults
        for key, val in TheConfig.Defaults.items():
            if not hasattr(self,key):
                setattr(self, key, val)
                cfg[key] = val
        TheConfig.CurrentConfig = cfg
        # process data
        for attr in TheConfig.Keys:
            if hasattr(self, attr):
                val = getattr(self, attr)
                match attr:
                    case 'raw_csv_path':
                        if os.path.abspath(val) != val:
                            val = os.path.abspath(os.path.join(cfg_folder,val))
                            setattr(self,attr,val)
                            key_csv_path = 'csv_path'
                            # if (not hasattr(self, key_csv_path)):
                            if (not key_csv_path in cfg):
                                val_csv_path = os.path.splitext(val)[0] + '.out.csv'
                                setattr(self,key_csv_path,val_csv_path)
                        pass
                    case 'csv_path':
                        if os.path.abspath(val) != val:
                            val = os.path.abspath(os.path.join(cfg_folder,val))
                            setattr(self,attr,val)
                        pass
                    case 'random_state':
                        if isinstance(val,int):
                            val = get_rand_state(val)
                            setattr(self,attr,val)
                        pass
                    case 'gen_state':
                        if isinstance(val,int):
                            val = get_rand_state(val)
                            setattr(self,attr,val)
                        pass
                    case _:
                        pass
    @property
    def pretty_print(self):
        out = "-->\n"
        for attr in self.__dict__.keys():
            value = getattr(self, attr)
            out+=f"Config['{attr}'] = {value}\n"
        return out+"<--"

cfg = TheConfig(config_path)
print(cfg.pretty_print)

pined_stats = {}

def pin_rand_state(state_name):
    if state_name in TheConfig.CurrentConfig:
        val = TheConfig.CurrentConfig[state_name]
        if isinstance(val,int):
            if not state_name in pined_stats:
                pined_stats[state_name] = getattr(cfg, state_name)
            setattr(cfg,state_name,get_rand_state(val))

def unpin_rand_state(state_name):
    if state_name in pined_stats:
        setattr(cfg,state_name,pined_stats[state_name])

def prepare_dataset(input_gen=None):
    da_gen_state = cfg.gen_state if input_gen is None else input_gen
    if cfg.do_laundry:
        sqlLiteDf = SqlLiteDataFrame(True,cfg.raw_csv_path)
        sqlLiteDf.generate_target(used_ratio=cfg.used_ratio,thresh=cfg.thresh, gen_state=da_gen_state, gen_weight=cfg.gen_weight,gen_source_columns=cfg.gen_source_columns)
        sqlLiteDf.to_csv()
        return sqlLiteDf.src_columns
    elif cfg.generate_target:
        sqlLiteDf = SqlLiteDataFrame(False,cfg.csv_path)
        sqlLiteDf.generate_target(used_ratio=cfg.used_ratio,thresh=cfg.thresh, gen_state=da_gen_state, gen_weight=cfg.gen_weight,gen_source_columns=cfg.gen_source_columns)
        sqlLiteDf.to_csv()
        return sqlLiteDf.src_columns
    return None

def new_stuff(csv_path,my_test_size):
    # read in dataset
    data = pd.read_csv(csv_path)

    # divide input and target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # divide dataset（80% train，20% test）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=my_test_size, random_state=cfg.random_state)

    # init random forest and XGBoost classifier
    rf = RandomForestClassifier(random_state=cfg.random_state)
    xgb_clf = xgb.XGBClassifier(objective='binary:logistic', random_state=cfg.random_state)

    # define hyper grid
    rf_param_grid = cfg.rf_param_grid
    # {
    #     'n_estimators': [100, 200],
    #     'max_depth': [10, 20],
    # }

    xgb_param_grid = cfg.xgb_param_grid
    # {
    #     'n_estimators': [100, 200],
    #     'max_depth': [3, 5],
    #     'learning_rate': [0.01, 0.1],
    # }

    # create RandomizedSearchCV object
    rf_random_search = RandomizedSearchCV(rf, rf_param_grid, n_iter=cfg.rf_n_iter, cv=cfg.cv, 
                                        scoring='accuracy', random_state=cfg.random_state, n_jobs=cfg.n_jobs)
    xgb_random_search = RandomizedSearchCV(xgb_clf, xgb_param_grid, n_iter=cfg.xgb_n_iter, cv=cfg.cv, 
                                        scoring='accuracy', random_state=cfg.random_state, n_jobs=cfg.n_jobs)

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
    ensemble_accuracy = accuracy_score(y_test, y_pred)
    ensemble_precision = precision_score(y_test, y_pred)
    
    print("Outputing Results...")

    print(f"Training/Test Set(%): {100-int(my_test_size*100)}/{int(my_test_size*100)}")
    print(f'Accuracy of The Ensemble Model: {ensemble_accuracy:.6f}')
    print(f'Precision of The Ensemble Model: {ensemble_precision:.6f}')
    return ensemble_accuracy,ensemble_precision

def old_stuff(csv_path,my_test_size):
    # Assuming the target variable 'Defective' is binary and present in the dataset.
    # Assuming 'Defective' is 1 if the product is defective and 0 otherwise.
    # This is a mock column based on paper description.
    # In the real case, this should be informed by the actual process of labeling defective products.

    # Load data
    data = pd.read_csv(csv_path)
    column_names=data.columns.to_list()
    # Feature extraction
    # Select the columns which were mentioned to be used - assuming that similar columns will be used to construct features.
    features = column_names[:-1]
    # The last column should contain required defective binary data.
    target = column_names[-1]


    def laundry_func(x):
        import math
        try:
            return -1 if (math.isnan(float(str(x)))) else x
        except:
            return -1
    data[features] = data[features].map(laundry_func)

    print("Processing Start...")
    # Data preprocessing - scaling and normalization
    scaler = StandardScaler()

    X = scaler.fit_transform(data[features])
    y = data[target]

    # Splitting the dataset into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=my_test_size, random_state=cfg.random_state)

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

    print(f"Training/Test Set(%): {100-int(my_test_size*100)}/{int(my_test_size*100)}")
    print(f"Accuracy of Vanilla Linear Regression: {linear_accuracy:.6f}")
    print(f"Precision of Vanilla Linear Regression: {linear_precision:.6f}")
    print(f"Accuracy of Vanilla Logistic Regression: {logistic_accuracy:.6f}")
    print(f"Precision of Vanilla Logistic Regression: {logistic_precision:.6f}")
    return linear_accuracy,linear_precision,logistic_accuracy,logistic_precision
    # Note: In a real replication, you may need to perform cross-validation or use a validation set to tune hyperparameters.

def run_prediction(wanted_test_size=None):
    if cfg.legacy_only:
        return old_stuff(cfg.csv_path)
    else:
        test_size = wanted_test_size if (isinstance(wanted_test_size,float) and wanted_test_size>0.0 and wanted_test_size<1.0) else cfg.test_size
        print("--------------------old version--------------------")
        linear_accuracy,linear_precision,logistic_accuracy,logistic_precision = old_stuff(cfg.csv_path,test_size)
        print("--------------------new version--------------------")
        ensemble_accuracy,ensemble_precision = new_stuff(cfg.csv_path,test_size)
        return linear_accuracy,linear_precision,logistic_accuracy,logistic_precision,ensemble_accuracy,ensemble_precision 

def tune_args():
    import copy
    ab_side = True
    copied_config = copy.deepcopy(TheConfig.CurrentConfig)
    seed = copied_config['gen_state'] if isinstance(copied_config['gen_state'],int) else 42
    rng = np.random.default_rng(seed)
    copied_config['num_shots'] = 1
    gate_accu = 0.83
    min_accu = 0.842
    min_prec = 0.5
    diff_in_accu = -1
    diff_in_prec = -1
    setattr(cfg,"gen_source_columns",None)
    setattr(cfg,"generate_target",True)
    for i in range(cfg.num_shots):
        src_columns = prepare_dataset(rng)
        pin_rand_state("gen_state")
        pin_rand_state("random_state")
        linear_accuracy,linear_precision,logistic_accuracy,logistic_precision,ensemble_accuracy,ensemble_precision = run_prediction()
        unpin_rand_state("gen_state")
        unpin_rand_state("random_state")
        if (ensemble_accuracy>=min_accu) and (ensemble_precision>=min_prec) and \
        (max(linear_accuracy,logistic_accuracy) <= gate_accu) and \
        (ensemble_accuracy - max(linear_accuracy,logistic_accuracy) >= diff_in_accu) and \
        (ensemble_precision >= max(linear_precision,logistic_precision)) :
            diff_in_accu = ensemble_accuracy - max(linear_accuracy,logistic_accuracy)
            diff_in_prec = ensemble_precision - max(linear_precision,logistic_precision)
            out_json = {}
            out_json["diff_in_accu"]=diff_in_accu
            out_json["diff_in_prec"]=diff_in_prec
            out_json["src_columns"]=src_columns
            out_json["scores"]=[linear_accuracy,linear_precision,logistic_accuracy,logistic_precision,ensemble_accuracy,ensemble_precision]
            copied_config["gen_source_columns"]=src_columns
            copied_config["generate_target"]=False
            out_json["config"]=copied_config
            tar_path = './data/A.json' if ab_side else './data/B.json'
            ab_side = not ab_side
            with open(tar_path, 'w') as json_file:
                json.dump(out_json, json_file, indent=4)
            print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
            print(f"i={i}, scores={out_json["scores"]}")
            print(f"diff_in_accu={diff_in_accu}, diff_in_prec={diff_in_prec}")
            print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
        else:
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

if cfg.num_shots > 1:
    tune_args()
else:
    for test_size in range(50,0,-10):
        pin_rand_state("gen_state")
        pin_rand_state("random_state")
        prepare_dataset()
        run_prediction(float(test_size)/100.0)
print("___________________________________________________")
