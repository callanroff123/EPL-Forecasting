###############################
### Executes E2E ML pipeline ##
###############################

# Import libraries/modules
from app.data_prep.etl import run_etl_pipeline
from app.model.train import fetch_epl_data
from app.data_prep.prediction_data import get_prediction_data
from app.model.train import train_and_save_standard_scaler
from app.model.train import save_best_xgboost_model
from app.model.predict import xgboost_preds
from app.config import X_COLS


# Function executing each step in the end-to-end pipeline
def training_pipeline():
    run_etl_pipeline()
    df = fetch_epl_data()
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


if __name__ == "__main__":
    print("Hello human! I am an AI which predicts the result of an upcoming EPL fixture.")
    home_team = input("Please specify the home team of this fixture: ")
    away_team = input("Please specify the away team of this fixture: ")
    get_predictions(home_team = home_team, away_team = away_team)


get_prediction_data("Fulham", "Tottenham")


