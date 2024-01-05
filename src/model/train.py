################################################################################################
### In this version, we want to build a model which classifies a home-team win, draw or loss ###
################################################################################################


# Import required libraries
import warnings
import multiprocessing as mp
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import math
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score
from sklearn.utils.class_weight import compute_sample_weight
from sqlalchemy import create_engine
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import os
import uuid
from src.config import INPUT_PATH, MODEL_PATH, X_COLS


# Read in data from the ETL step
def fetch_epl_data(source = "csv"):
    if source == "csv":
        df = pd.read_csv(str(INPUT_PATH) + "/epl_data.csv")
        return(df.dropna(axis = 0).reset_index(drop = True))
    elif source == "postgre_db":
        connection_params = json.load(
            open("/Users/callanroff/Desktop/Acc. Keyzzz/postgresql_conn_params.json", "r")
        )
        schema_name = "web_scraping"
        table_name = "epl_data"
        conn = psycopg2.connect(
            host = connection_params["host"],
            database = connection_params["database"],
            user = connection_params["user"],
            password = connection_params["password"]
        )
        df = pd.read_sql(
            f'''
                SELECT *
                FROM {schema_name}.{table_name}
            ''',
            con = conn
        )
        conn.close()
        return(df.dropna(axis = 0).reset_index(drop = True))
    else:
        raise ValueError("Only 'csv' and 'postgre_db' allowed for the source argument")
    

# Build a standard scaler model which is fitted on the epl data
# When we want to make a prediction later on, we call this same scalar
def train_and_save_standard_scaler(df, X_cols):
    X = df[X_cols]
    scaler = StandardScaler()
    scaler.fit(X)
    with open(str(MODEL_PATH) + "/scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)


# Transform data with the fitted standerad scaler
def transform_standard_scaler(df, X_cols):
    X = df[X_cols]
    with open(str(MODEL_PATH) + "/scaler.pkl", 'rb') as file:  
        scaler = pickle.load(file)
    X_scaled = pd.DataFrame(
        scaler.transform(X), 
        columns = [col + "_SCALED" for col in X.columns]
    )
    return(X_scaled)


# PCA step for diimensionality reduction, with the standard-scaled feature space as input
def train_and_save_pca(X_scaled):
    pca = PCA(
        svd_solver = "full",
        n_components = 0.95
    )
    pca.fit(X_scaled)
    with open(str(MODEL_PATH) + "/pca.pkl", "wb") as file:
        pickle.dump(pca, file)


# Implement a standard-scalar transformation across the feature space
# Also choose what quantity we are predicting:
# * "target = base_result" -> We are predicting either home win, home draw or home loss
# * "target = home_win" -> Binary output: 1 if home team win, else 0
# * "target = min_goals" -> Also binary output: 1 if at least 3 goals are scored, else 0
def preprocess_data(df, X_cols, pca = False, target = "base_result"):
    X = df[X_cols]
    with open(str(MODEL_PATH) + "/scaler.pkl", 'rb') as file:  
        scaler = pickle.load(file)
    X_scaled = pd.DataFrame(
        scaler.transform(X), 
        columns = [col + "_SCALED" for col in X.columns]
    )
    if pca == True:
        with open(str(MODEL_PATH) + "/pca.pkl", "rb") as file:
            pca = pickle.load(file)
        X_scaled_pca = pd.DataFrame(pca.transform(X_scaled))
        dat_new = X_scaled_pca.join(df[["RESULT", "TOT_GOALS"]])
    else:
        dat_new = X_scaled.join(df[["RESULT", "TOT_GOALS"]])
    if target == "base_result":
        dat_new["RESULT"] = dat_new["RESULT"].apply(
            lambda x: 0 if x == "Home Loss" else (1 if x == "Draw" else 2)
        )
    elif target == "home_win":
        dat_new["RESULT"] = dat_new["RESULT"].apply(
            lambda x: 0 if x != "Home Win" else 1
        )
    elif target == "min_goals":
        dat_new["RESULT"] = dat_new["TOT_GOALS"].apply(
            lambda x: 0 if x < 3 else 1
        )
    else:
        raise ValueError("Only 'base_result', 'home_win' and 'min_goals' allowed for 'target' argument.")
    dat_new.drop("TOT_GOALS", axis = 1, inplace = True)
    return(dat_new)


# Implement XGBoost classifier based on a series hyperparameter inputs
# Extract the combination of hyperparamaters which results in the lowest CV error (on the training set)
def xgboost_grid_search(df, X_cols, pca = False, target = "base_result"):
    df_clean = preprocess_data(df, X_cols, target = target, pca = pca)
    X = df_clean.iloc[:, :-1]
    y = df_clean.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 101)
    max_depth_vals = [5, 10]
    learning_rate_vals = [0.1, 0.05]
    hyperparam_grid = {
        #"min_child_weight": [1, 3],
        #"learning_rate": [0.01, 0.05],
        "max_depth": max_depth_vals,
        "learning_rate": learning_rate_vals
    }
    if target == "base_result":
        model = XGBClassifier(
            objective = "multi:softprob", 
            num_classes = 3, 
            class_weight = 'balanced',
            n_estimators = 250,
            colsample_bytree = 1
        )
        grid = GridSearchCV(
            estimator = model,
            param_grid = hyperparam_grid,
            cv = 10,
            n_jobs = 2,
            scoring = "accuracy"
        )
        grid.fit(X_train, y_train)
    else:
        model = XGBClassifier(
            objective = "binary:logistic",
            n_estimators = 250,
            colsample_bytree = 1
        )
        grid = GridSearchCV(
            estimator = model,
            param_grid = hyperparam_grid,
            cv = 10,
            n_jobs = 2,
            scoring = "accuracy"
        )
        grid.fit(X_train, y_train)
    best_params = grid.best_params_
    y_pred = grid.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average = "weighted")
    prec = precision_score(y_test, y_pred, average = "weighted")
    rec = recall_score(y_test, y_pred, average = "weighted")
    out_dict = {
        "model": "XGBoost Classifier",
        "num_classes": len(set(list(y_test) + list(y_pred))),
        "standard_scaler_transform": True,
        "pca_transform": pca,
        "params_tested": hyperparam_grid,
        "best_parameters": best_params,
        "accuracy": acc,
        "weighted_f1_score": f1,
        "weighted_precision": prec,
        "weighted_recall": rec,
        "run_id": str(uuid.uuid1()),
        "run_date": str(datetime.now())
    }
    with open(str(MODEL_PATH) + "/model_diagnostics.json", "w") as file:
        json.dump(out_dict, file)
    return(out_dict)


# Save the best performing model (i.e., best hyper-parameter combinations)
def save_best_xgboost_model(df, X_cols, pca = False, target = "base_result"):
    df_scaled = preprocess_data(df, X_cols, pca = pca, target = target)
    X = df_scaled.iloc[:, :-1]
    y = df_scaled.iloc[:, -1]
    hyperparameters = xgboost_grid_search(df, X_cols)["best_parameters"]
    if target == "base_result":
        model = XGBClassifier(
            objective = "multi:softprob", 
            num_classes = 3,
            class_weight = 'balanced',
            n_jobs = 3,
            colsample_bytree = 1,
            learning_rate = hyperparameters["learning_rate"],
            n_estimators = 250,
            max_depth = hyperparameters["max_depth"]
        )
        model.fit(X, y)
    else:
        model = XGBClassifier(
            objective = "binary:logistic",
            n_jobs = 3,
            colsample_bytree = 1,
            learning_rate = hyperparameters["learning_rate"],
            n_estimators = 250,
            max_depth = hyperparameters["max_depth"]
        )
        model.fit(X, y)
    with open(str(MODEL_PATH) + "/model.pkl", "wb") as file:
        pickle.dump(model, file)