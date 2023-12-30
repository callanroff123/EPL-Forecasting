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
from sqlalchemy import create_engine
import psycopg2
import json
import pickle
import os
import uuid
from app.config import INPUT_PATH, OUTPUT_PATH, MODEL_PATH, X_COLS


# Fit the parameter-tuned XGBoost regressor on new data
def xgboost_preds(new_df, new_X, X_cols):
    with open(str(MODEL_PATH) + '/model.pkl', 'rb') as file:  
        model = pickle.load(file)
    y_forecast = model.predict(new_X)
    new_df["RESULT_PREDICTION"] = y_forecast
    return(new_df)


# Push forecasts to output file path as csv file
def push_xgboost_forecast_to_csv(pred_df):
    pred_df.to_csv(str(OUTPUT_PATH) + "/epl_predictions.csv", encoding = "utf-8", index = False)


# Push forecasts to PostgreSQL DB
def push_xgboost_forecast_to_db(pred_df, connection_params, schema_name, table_name):
    conn = psycopg2.connect(
        host = connection_params["host"],
        database = connection_params["database"],
        user = connection_params["user"],
        password = connection_params["password"]
    )
    engine = create_engine('postgresql://' + connection_params["user"] + ':' + connection_params["password"] + '@' + connection_params["host"] + ':5432/' + connection_params["database"])
    df = pred_df
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