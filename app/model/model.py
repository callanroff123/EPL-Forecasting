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
import os
import uuid


# Read in data from the ETL step
def fetch_epl_data():
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
    return(df)


# Implement a standard-scalar transformation across the feature space
def standardize_data(df, X_cols):
    X = df[X_cols]
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), 
        columns = [col + "_SCALED" for col in X.columns]
    )
    dat_new = X_scaled.join(df[["RESULT"]])
    dat_new["RESULT"] = dat_new["RESULT"].apply(
        lambda x: 0 if x == "Home Loss" else (1 if x == "Draw" else 2)
    )
    return(dat_new)


# Implement XGBoost classifier based on a series hyperparameter inputs
# Extract the combination of hyperparamaters which results in the lowest CV error (on the training set)
def xgboost_grid_search(df, X_cols):
    df = standardize_data(df, X_cols)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 101)
    hyperparam_grid = {
        "min_child_weight": [1, 3],
        "learning_rate": [0.01, 0.05],
        "max_depth": [5, 10]
    }
    model = XGBClassifier(
        objective = "multi:softprob", 
        num_classes = 3, 
        n_estimators = 500,
        colsample_bytree = 1
    )
    grid = GridSearchCV(
        estimator = model,
        param_grid = hyperparam_grid,
        cv = 5,
        n_jobs = 3
    )
    sample_weights = compute_sample_weight(
        class_weight = 'balanced',
        y = y_train 
    )
    grid.fit(X_train, y_train, sample_weight = sample_weights)
    best_params = grid.best_params_
    y_pred = grid.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    out_dict = {
        "best_parameters": best_params,
        "accuracy": acc,
        "run_id": uuid.uuid1(),
        "run_date": datetime.now()
    }
    return(out_dict)


# Implement RF classifier
def rf_function():
    pass


# Fit the parameter-tuned XGBoost regressor on new data
def xgboost_preds(df, new_df, X_cols):
    df_scaled = standardize_data(df, X_cols)
    new_X = standardize_data(new_df, X_cols)
    X = df_scaled.iloc[:, :-1]
    y = df_scaled.iloc[:, -1]
    hyperparameters = xgboost_grid_search(df, X_cols)["best_parameters"]
    model = XGBClassifier(
        objective = "multi:softprob", 
        num_classes = 3,
        n_jobs = 3,
        min_child_weight = hyperparameters["min_child_weight"],
        colsample_bytree = 1,
        learning_rate = hyperparameters["learning_rate"],
        n_estimators = 500,
        max_depth = hyperparameters["max_depth"]
    )
    sample_weights = compute_sample_weight(
        class_weight = 'balanced',
        y = y
    )
    model.fit(X, y)
    y_forecast = model.predict(new_X)
    new_df["RESULT_PREDICTION"] = y_forecast
    new_df["RESULT_PREDICTION"] = new_df["RESULT_PREDICTION"].apply(
        lambda x: "Home Win" if x == 2 else ("Draw" if x == 1 else "Home Loss")
    )
    return(new_df)


# Push forecasts to PostgreSQL DB
def push_xgboost_forecast_to_db(connection_params, schema_name, table_name):
    conn = psycopg2.connect(
        host = connection_params["host"],
        database = connection_params["database"],
        user = connection_params["user"],
        password = connection_params["password"]
    )
    engine = create_engine('postgresql://' + connection_params["user"] + ':' + connection_params["password"] + '@' + connection_params["host"] + ':5432/' + connection_params["database"])
    df = xgboost_preds()
    conn.autocommit = True
    cursor = conn.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS {schema_name}.{table_name}")
    cursor.execute(
        f'''
            CREATE TABLE {schema_name}.{table_name} (
                HOME_TEAM VARCHAR(100),
                AWAY_TEAM VARCHAR(100),
                SEASON VARCHAR(100),
                DATE DATE,
                DIFF_HA_LAST_5_HOME_WINS INT,
                DIFF_HA_LAST_5_HOME_DRAWS INT,
                DIFF_HA_LAST_5_HOME_LOSSES INT,
                DIFF_HA_LAST_5_AWAY_WINS INT,
                DIFF_HA_LAST_5_AWAY_DRAWS INT,
                DIFF_HA_LAST_5_AWAY_LOSSES INT,
                DIFF_HA_LAST_5_WINS INT,
                DIFF_HA_LAST_5_DRAWS INT, 
                DIFF_HA_LAST_5_LOSSES INT,
                DIFF_HA_GD_LAST_5 INT,
                DIFF_HA_PTS_THIS_SEASON INT,
                DIFF_HA_CURRENT_POSITION INT,
                DIFF_HA_SHOTS_LAST_5 INT,
                DIFF_HA_SHOTS_ON_TARGET_LAST_5 INT,
                HOME_TEAM_BIG_6_FLAG INT,
                AWAY_TEAM_BIG_6_FLAG INT,
                HOME_TEAM_PROMOTED_FROM_LAST_SEASON_FLAG INT,
                AWAY_TEAM_PROMOTED_FROM_LAST_SEASON_FLAG INT,
                SEASON_MONTH_NUM INT,
                HOME_TEAM_GAME_NUM INT,
                AWAY_TEAM_GAME_NUM INT,
                RESULT_PREDICTION VARCHAR(100),
            ); 
        '''
    )
    df.to_sql(
        table_name,
        engine,
        if_exists = "replace",
        schema = schema_name,
        index = False
    )
    conn.close()


# Execute model development pipeline
def run_model_pipeline():
    pass


df = fetch_epl_data()
X_cols = [
    "DIFF_HA_LAST_5_HOME_WINS",
    "DIFF_HA_LAST_5_HOME_DRAWS",
    "DIFF_HA_LAST_5_HOME_LOSSES",
    "DIFF_HA_LAST_5_AWAY_WINS",
    "DIFF_HA_LAST_5_AWAY_DRAWS",
    "DIFF_HA_LAST_5_AWAY_LOSSES",
    "DIFF_HA_LAST_5_WINS",
    "DIFF_HA_LAST_5_DRAWS", 
    "DIFF_HA_LAST_5_LOSSES",
    "DIFF_HA_GD_LAST_5",
    "DIFF_HA_PTS_THIS_SEASON",
    "DIFF_HA_CURRENT_POSITION",
    "DIFF_HA_SHOTS_LAST_5",
    "DIFF_HA_SHOTS_ON_TARGET_LAST_5",
    "HOME_TEAM_BIG_6_FLAG",
    "AWAY_TEAM_BIG_6_FLAG",
    "HOME_TEAM_PROMOTED_FROM_LAST_SEASON_FLAG",
    "AWAY_TEAM_PROMOTED_FROM_LAST_SEASON_FLAG",
    "SEASON_MONTH_NUM",
    "HOME_TEAM_GAME_NUM",
    "AWAY_TEAM_GAME_NUM"
]
standardize_data(df, X_cols)

