##########################################################################################################
### This file generates a prediction dataset #############################################################
### First we take in as input the user's fixture (simply the home/away team) #############################
### For those inputed teams, we find their most recent fixtures and extract those from the etl results ###
##########################################################################################################


# Import libraries/modules
import pandas as pd
import numpy as np
from src.config import INPUT_PATH, X_COLS


# Read in the results of the etl.py module
def read_in_epl_data():
    df = pd.read_csv(str(INPUT_PATH) + '/epl_data.csv')
    return(df)


# Get the prediction dataset for a given home/away team
def get_prediction_data(home_team, away_team):
    df = read_in_epl_data()
    current_pl_teams = list(set(list(df[df["SEASON"] == max(df["SEASON"])]["HOME_TEAM"]) + list(df[df["SEASON"] == max(df["SEASON"])]["AWAY_TEAM"])))
    if (home_team not in current_pl_teams) or (away_team not in current_pl_teams):
        raise ValueError("Home or Away Team Selected not in Current PL Season :/")
    elif home_team == away_team:
        raise ValueError("Home and Away Teams Can't be the Same :/")
    else:
        out_df_raw = pd.DataFrame()
        for x in ["HOME", "AWAY"]:
            latest_team_dates = df[df[f"{x}_TEAM"].isin(current_pl_teams)].groupby(f"{x}_TEAM").agg("max").reset_index(drop = False)[[f"{x}_TEAM", "DATE"]]
            df_latest = pd.merge(
                left = df,
                right = latest_team_dates,
                on = [
                    f"{x}_TEAM",
                    "DATE"
                ],
                how = "inner"
            )
            df_latest["FOCUS_TEAM"] = df_latest[f"{x}_TEAM"]
            df_latest["FOCUS_TEAM_HA"] = x
            out_df_raw = pd.concat([out_df_raw, df_latest], axis = 0).reset_index(drop = True)
        latest_team_dates = out_df_raw.groupby("FOCUS_TEAM").agg("max").reset_index(drop = False)[["FOCUS_TEAM", "DATE"]]
        out_df_clean = pd.merge(
            left = out_df_raw,
            right = latest_team_dates,
            on = [
                "FOCUS_TEAM",
                "DATE"
            ],
            how = "inner"
        )
        new_df = pd.DataFrame()
        for x in ["HOME", "AWAY"]:
            x_team_df = out_df_clean[out_df_clean["FOCUS_TEAM_HA"] == x][
                ["FOCUS_TEAM", "FOCUS_TEAM_HA"] + [
                    f"{x}_TEAM_LAST_5_" + y + "_" + z for y in ["HOME", "AWAY"] for z in ["WINS", "DRAWS", "LOSSES"]
                ] + [
                    f"{x}_TEAM_" + z + "_LAST_5" for z in ["WINS", "DRAWS", "LOSSES"]
                ] + [
                    f"{x}_TEAM_GD_LAST_5", 
                    f"{x}_TEAM_SHOTS_LAST_5",
                    f"{x}_TEAM_SHOTS_ON_TARGET_LAST_5",
                    f"{x}_TEAM_PTS", 
                    f"{x}_TEAM_POSITION",
                    f"{x}_TEAM_BIG_6_FLAG",
                    f"{x}_TEAM_PROMOTED_FROM_LAST_SEASON_FLAG",
                    f"{x}_TEAM_GAME_NUM"
                ]
            ].reset_index(drop = True)
            x_team_df = x_team_df.rename(columns = {
                f"{x}_TEAM_LAST_5_" + y + "_" + z: "LAST_5_" + y + "_" + z for y in ["HOME", "AWAY"] for z in ["WINS", "DRAWS", "LOSSES"]
            }).rename(columns = {
                f"{x}_TEAM_" + z + "_LAST_5": "LAST_5_" + z for z in ["WINS", "DRAWS", "LOSSES"]
            }).rename(columns = {
                f"{x}_TEAM_GD_LAST_5": "GD_LAST_5", 
                f"{x}_TEAM_SHOTS_LAST_5": "SHOTS_LAST_5",
                f"{x}_TEAM_SHOTS_ON_TARGET_LAST_5": "SHOTS_ON_TARGET_LAST_5",
                f"{x}_TEAM_PTS": "PTS", 
                f"{x}_TEAM_POSITION": "POSITION",
                f"{x}_TEAM_BIG_6_FLAG": "BIG_6_FLAG",
                f"{x}_TEAM_PROMOTED_FROM_LAST_SEASON_FLAG": "PROMOTED_FROM_LAST_SEASON_FLAG",
                f"{x}_TEAM_GAME_NUM": "GAME_NUM"
            })
            new_df = pd.concat(
                [new_df, x_team_df],
                axis = 0
            )
        new_df["KEY"] = 0
        new_df = pd.merge(
            left = new_df,
            right = new_df,
            on = "KEY",
            how = "inner"
        )
        new_df = new_df[
            (
                (new_df["FOCUS_TEAM_x"] == home_team) &
                (new_df["FOCUS_TEAM_y"] == away_team)
            ) &
            (new_df["FOCUS_TEAM_x"] != new_df["FOCUS_TEAM_y"])
        ].reset_index(drop = True).iloc[[0]]
        new_df[f"DIFF_HA_GD_LAST_5"] = new_df[f"GD_LAST_5_x"] - new_df[f"GD_LAST_5_y"]
        new_df[f"DIFF_HA_SHOTS_LAST_5"] = new_df[f"SHOTS_LAST_5_x"] - new_df[f"SHOTS_LAST_5_y"]
        new_df[f"DIFF_HA_SHOTS_ON_TARGET_LAST_5"] = new_df[f"SHOTS_ON_TARGET_LAST_5_x"] - new_df[f"SHOTS_ON_TARGET_LAST_5_y"]
        new_df[f"DIFF_HA_PTS_THIS_SEASON"] = new_df[f"PTS_x"] - new_df[f"PTS_y"]
        new_df[f"DIFF_HA_CURRENT_POSITION"] = new_df[f"POSITION_x"] - new_df[f"POSITION_y"]
        new_df[f"HOME_TEAM_BIG_6_FLAG"] = new_df[f"BIG_6_FLAG_x"]
        new_df[f"AWAY_TEAM_BIG_6_FLAG"] = new_df[f"BIG_6_FLAG_y"]
        new_df[f"HOME_TEAM_PROMOTED_FROM_LAST_SEASON_FLAG"] = new_df[f"PROMOTED_FROM_LAST_SEASON_FLAG_x"]
        new_df[f"AWAY_TEAM_PROMOTED_FROM_LAST_SEASON_FLAG"] = new_df[f"PROMOTED_FROM_LAST_SEASON_FLAG_y"]
        new_df[f"HOME_TEAM_GAME_NUM"] = new_df[f"GAME_NUM_x"]
        new_df[f"AWAY_TEAM_GAME_NUM"] = new_df[f"GAME_NUM_y"]
        for z in ["WINS", "DRAWS", "LOSSES"]:
            new_df[f"DIFF_HA_LAST_5_{z}"] = new_df[f"LAST_5_{z}_x"] - new_df[f"LAST_5_{z}_y"]
            for y in ["HOME", "AWAY"]:
                new_df[f"DIFF_HA_LAST_5_{y}_{z}"] = new_df[f"LAST_5_{y}_{z}_x"] - new_df[f"LAST_5_{y}_{z}_y"]
    return(new_df[X_COLS])