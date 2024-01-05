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
from src.config import INPUT_PATH, MODEL_PATH, X_COLS
from src.data_prep.prediction_data import get_prediction_data


# Fit the parameter-tuned XGBoost regressor on new data
def xgboost_preds(home_team, away_team, pca = False):
    with open(str(MODEL_PATH) + '/model.pkl', 'rb') as file:  
        model = pickle.load(file)
    with open(str(MODEL_PATH) + '/scaler.pkl', 'rb') as file:  
        scaler = pickle.load(file)
    with open(str(MODEL_PATH) + '/pca.pkl', 'rb') as file:  
        pca = pickle.load(file)
    new_X = get_prediction_data(home_team = home_team, away_team = away_team)
    new_X_scaled = pd.DataFrame(
        scaler.transform(new_X),
        columns = [col + "_SCALED" for col in new_X.columns]
    )
    if pca == True:
        new_X_scaled_pca = pd.DataFrame(pca.transform(new_X_scaled))
        y_forecast = model.predict(new_X_scaled_pca)
    else:
        y_forecast = model.predict(new_X_scaled)
    return(y_forecast)