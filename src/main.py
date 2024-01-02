###############################
### Executes E2E ML pipeline ##
###############################


# venv not working for some reason :/ Hopefully not an issue when deployed in cloud
import sys
import os
sys.path.append("/Users/callanroff/Desktop/EPL-Forecasting")
os.environ["PYTHONPATH"]="/Users/callanroff/Desktop/EPL-Forecasting"


# Suppress SciPy NumPy version warning
import warnings
warnings.filterwarnings("ignore", category = Warning)


# Import libraries/modules
from src.data_prep.etl import run_etl_pipeline
from src.model.train import fetch_epl_data, train_and_save_standard_scaler, save_best_xgboost_model
from src.model.predict import xgboost_preds
from src.config import X_COLS


# Function executing each step in the end-to-end pipeline
def training_pipeline():
    print("Running ETL pipeline...")
    run_etl_pipeline()
    df = fetch_epl_data()
    print("Running ML training pipeline...")
    train_and_save_standard_scaler(df = df, X_cols = X_COLS)
    save_best_xgboost_model(df = df, X_cols = X_COLS)


# Function which gives us the predicted result based on 
def get_predictions(home_team, away_team):
    out = xgboost_preds(home_team = home_team, away_team = away_team)
    if out == 0:
        x = f"I predict the winner will be {away_team}."
    elif out == 1:
        x = f"I predict a draw."
    elif out == 2:
        x = f"I predict the winner will be {home_team}."
    else:
        raise ValueError("Something ain't right :/")
    return(x)


# Run script
if __name__ == "__main__":
    df = fetch_epl_data()
    teams = list(set(list(df[df["SEASON"] == max(df["SEASON"])]["HOME_TEAM"]) + list(df[df["SEASON"] == max(df["SEASON"])]["AWAY_TEAM"])))
    print("Hello human! I am an AI which predicts the result of an upcoming EPL fixture.")
    run_mod = input("Do you want to refresh the model (y/n)? Note 'y' for this option may take 5-10 minutes: ")
    if run_mod == "y":
        print("Executing training pipeline...")
        training_pipeline()
        print("Model successfully updated!")
    elif run_mod == "n":
        pass
    else:
        input("Invalid input. Only 'y' for yes or 'n' for no allowed: ")
    again = "y"
    while True:
        home_team = input("Please specify the home team of this fixture: ")
        away_team = input("Please specify the away team of this fixture: ")
        while (home_team not in teams) or (away_team not in teams):
            print("Make sure the teams you input are from the following list: ")
            print(teams)
            home_team = input("Please specify the home team of this fixture: ")
            away_team = input("Please specify the away team of this fixture: ")
        print(get_predictions(home_team = home_team, away_team = away_team))
        again = input("Predict another fixture (y/n)? ")
        if again == "n":
            break
        elif again == "y":
            pass
        else:
            print("Only 'y' for yes or 'n' for no allowed")
    print("See ya later human!")