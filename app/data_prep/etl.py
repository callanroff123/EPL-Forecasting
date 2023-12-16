# Import required libraries
import psycopg2
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import json
import time
from datetime import datetime
from datetime import timedelta
import math


# Automated download of all CSV files from www.football-data.co.uk
def get_raw_historical_epl_data():
    df_raw = pd.DataFrame()
    for season in ['1112', '1213', '1314', '1415', '1516', '1617', '1718', '1819', '1920', '2021', '2122', '2223']:
        df_season = pd.read_csv("https://www.football-data.co.uk/mmz4281/" + season + "/E0.csv")
        df_season["Season"] = "20" + season[0:2] + "/20" + season[2:4] 
        df_raw = pd.concat([df_raw, df_season]).reset_index(drop = True)
    return(df_raw)


# Clean the extracted EPL data and perform some basic feature engineering
def clean_raw_epl_data():
    df_raw = get_raw_historical_epl_data()
    df_raw["SEASON"] = df_raw["Season"]
    df_raw["HOME_TEAM"] = df_raw["HomeTeam"]
    df_raw["AWAY_TEAM"] = df_raw["AwayTeam"]
    df_raw["Date"] = df_raw["Date"].astype(str)
    df_raw = df_raw[df_raw["Date"].str[0].isin(["0", "1", "2", "3"])]
    df_raw["DATE"] = pd.to_datetime(
        df_raw["Date"].apply(
            lambda x: 
                str(x)[-4:] + '-' + str(x)[-7:-5] + '-' + str(x)[0:2] if len(str(x)) == 10 else "20" + str(x)[-2:] + '-' + str(x)[-5:-3] + '-' + str(x)[0:2]        
        )
    )
    df_raw["HOUR"] = df_raw["Time"].str[0:2]
    df_raw["REFEREE"] = df_raw["Referee"]
    df_raw["DIFF_HA_FT_GOALS"] = df_raw["FTHG"] - df_raw["FTAG"]
    df_raw["DIFF_HA_HT_GOALS"] = df_raw["HTHG"] - df_raw["HTAG"]
    df_raw["DIFF_HA_SHOTS"] = df_raw["HS"] - df_raw["AS"]
    df_raw["DIFF_HA_SHOTS_ON_TARGET"] = df_raw["HST"] - df_raw["AST"]
    df_raw["DIFF_HA_CORNERS"] = df_raw["HC"] - df_raw["AC"]
    df_raw["DIFF_HA_FOULS"] = df_raw["HF"] - df_raw["AF"]
    df_raw["DIFF_HA_YELLOW_CARDS"] = df_raw["HY"] - df_raw["AY"]
    df_raw["DIFF_HA_RED_CARDS"] = df_raw["HR"] - df_raw["AR"]
    df_raw["AVG_ODDS_H_WIN"] = df_raw["AvgH"]
    df_raw["AVG_ODDS_A_WIN"] = df_raw["AvgA"]
    df_raw["AVG_ODDS_DRAW"] = df_raw["AvgD"]
    df_raw["AVG_H_WIN"] = df_raw["AvgH"]
    df_raw["AVG_ODDS_MORE_THAN_2_GOALS"] = df_raw["Avg>2.5"]
    df_raw["AVG_ODDS_2_OR_LESS_GOALS"] = df_raw["Avg<2.5"]
    df_raw["RESULT"] = df_raw["FTR"].apply(
        lambda x: "Home Win" if x == "H" else (
            "Draw" if x == "D" else "Home Loss"
        )
    )
    return(df_raw)


# Perform advanced feature engineering (detailed home and away form)
def epl_feature_engineering_1():
    df = clean_raw_epl_data()
    df["HOME_WIN_FLAG"] = df["RESULT"].apply(
        lambda x: 1 if x == "Home Win" else 0
    )
    df["DRAW_FLAG"] = df["RESULT"].apply(
        lambda x: 1 if x == "Draw" else 0
    )
    df["HOME_LOSS_FLAG"] = df["RESULT"].apply(
        lambda x: 1 if x == "Home Loss" else 0
    )
    df_new_1 = pd.DataFrame()
    df_new_2 = pd.DataFrame()
    for home_team in df["HOME_TEAM"].unique():
        df_home_team = df[df["HOME_TEAM"] == home_team]
        df_home_team = df_home_team.sort_values("DATE", ascending = True).reset_index(drop = True)
        for season in df_home_team["SEASON"].unique():
            df_home_team_season = df_home_team[df_home_team["SEASON"] == season].sort_values("DATE", ascending = True).reset_index(drop = True)
            df_home_team_season["HOME_TEAM_LAST_5_HOME_WINS"] = df_home_team_season["HOME_WIN_FLAG"].rolling(5).sum()
            df_home_team_season["HOME_TEAM_LAST_5_HOME_DRAWS"] = df_home_team_season["DRAW_FLAG"].rolling(5).sum()
            df_home_team_season["HOME_TEAM_LAST_5_HOME_LOSSES"] = df_home_team_season["HOME_LOSS_FLAG"].rolling(5).sum()
            df_home_team_season["HOME_TEAM_LAST_5_HOME_WINS"] = df_home_team_season["HOME_TEAM_LAST_5_HOME_WINS"].shift(1)
            df_home_team_season["HOME_TEAM_LAST_5_HOME_DRAWS"] = df_home_team_season["HOME_TEAM_LAST_5_HOME_DRAWS"].shift(1)
            df_home_team_season["HOME_TEAM_LAST_5_HOME_LOSSES"] = df_home_team_season["HOME_TEAM_LAST_5_HOME_LOSSES"].shift(1)
            df_new_1 = pd.concat([df_new_1, df_home_team_season]).sort_values(["DATE", "HOUR"]).reset_index(drop = True)
    for away_team in df["AWAY_TEAM"].unique():
        df_away_team = df[df["AWAY_TEAM"] == away_team]
        df_away_team = df_away_team.sort_values("DATE", ascending = True).reset_index(drop = True)
        for season in df_away_team["SEASON"].unique():
            df_away_team_season = df_away_team[df_away_team["SEASON"] == season].sort_values("DATE", ascending = True).reset_index(drop = True)
            df_away_team_season["AWAY_TEAM_LAST_5_AWAY_WINS"] = df_away_team_season["HOME_LOSS_FLAG"].rolling(5).sum()
            df_away_team_season["AWAY_TEAM_LAST_5_AWAY_DRAWS"] = df_away_team_season["DRAW_FLAG"].rolling(5).sum()
            df_away_team_season["AWAY_TEAM_LAST_5_AWAY_LOSSES"] = df_away_team_season["HOME_WIN_FLAG"].rolling(5).sum()
            df_away_team_season["AWAY_TEAM_LAST_5_AWAY_WINS"] = df_away_team_season["AWAY_TEAM_LAST_5_AWAY_WINS"].shift(1)
            df_away_team_season["AWAY_TEAM_LAST_5_AWAY_DRAWS"] = df_away_team_season["AWAY_TEAM_LAST_5_AWAY_DRAWS"].shift(1)
            df_away_team_season["AWAY_TEAM_LAST_5_AWAY_LOSSES"] = df_away_team_season["AWAY_TEAM_LAST_5_AWAY_LOSSES"].shift(1)
            df_away_team_season_final = df_away_team_season[[
                "SEASON", 
                "HOME_TEAM", 
                "AWAY_TEAM", 
                "DATE", 
                "AWAY_TEAM_LAST_5_AWAY_WINS",
                "AWAY_TEAM_LAST_5_AWAY_DRAWS",
                "AWAY_TEAM_LAST_5_AWAY_LOSSES"
            ]]
            df_new_2 = pd.concat([df_new_2, df_away_team_season_final]).sort_values(["DATE"]).reset_index(drop = True)
    df_new = pd.merge(
        left = df_new_1, 
        right = df_new_2,
        on = [
            "SEASON", 
            "HOME_TEAM", 
            "AWAY_TEAM", 
            "DATE"
        ],
        how = "inner"
    )
    df_new_partial_1 = df_new[[
        "HOME_TEAM",
        "AWAY_TEAM",
        "DATE",
        "HOME_TEAM_LAST_5_HOME_WINS",
        "HOME_TEAM_LAST_5_HOME_DRAWS",
        "HOME_TEAM_LAST_5_HOME_LOSSES"
    ]]
    df_new_partial_2 = df_new[[
        "HOME_TEAM",
        "AWAY_TEAM",
        "DATE",
        "AWAY_TEAM_LAST_5_AWAY_WINS",
        "AWAY_TEAM_LAST_5_AWAY_DRAWS",
        "AWAY_TEAM_LAST_5_AWAY_LOSSES"
    ]]
    df_home_away_merge = df_new_partial_1.merge(
        right = df_new_partial_2,
        left_on = "HOME_TEAM",
        right_on = "AWAY_TEAM",
        how = "inner"
    )
    df_home_away_merge["DATE_DIFF"] = (df_home_away_merge["DATE_x"] - df_home_away_merge["DATE_y"]).dt.days
    df_home_away_merge_1 = df_home_away_merge[[
        "HOME_TEAM_x",
        "AWAY_TEAM_x",
        "DATE_x",
        "DATE_y",
        "AWAY_TEAM_LAST_5_AWAY_WINS",
        "AWAY_TEAM_LAST_5_AWAY_DRAWS",
        "AWAY_TEAM_LAST_5_AWAY_LOSSES",
        "DATE_DIFF"
    ]]
    df_home_away_merge_1 = df_home_away_merge_1[df_home_away_merge_1["DATE_DIFF"] >= 0]
    df_home_away_merge_1["ROW_NUM"] = df_home_away_merge_1.sort_values("DATE_DIFF", ascending = True).groupby(["HOME_TEAM_x", "DATE_x"]).cumcount() + 1
    df_home_away_merge_1_final = df_home_away_merge_1[df_home_away_merge_1["ROW_NUM"] == 1].reset_index(drop = True)
    df_home_away_merge_1_final = df_home_away_merge_1_final.rename(columns = {
        "HOME_TEAM_x": "HOME_TEAM",
        "DATE_x": "DATE",
        "AWAY_TEAM_LAST_5_AWAY_WINS": "HOME_TEAM_LAST_5_AWAY_WINS",
        "AWAY_TEAM_LAST_5_AWAY_DRAWS": "HOME_TEAM_LAST_5_AWAY_DRAWS",
        "AWAY_TEAM_LAST_5_AWAY_LOSSES": "HOME_TEAM_LAST_5_AWAY_LOSSES"
    })
    df_home_away_merge_1_final = df_home_away_merge_1_final[[
        "HOME_TEAM",
        "DATE",
        "HOME_TEAM_LAST_5_AWAY_WINS",
        "HOME_TEAM_LAST_5_AWAY_DRAWS",
        "HOME_TEAM_LAST_5_AWAY_LOSSES"
    ]]
    df_new = pd.merge(
        left = df_new, 
        right = df_home_away_merge_1_final,
        on = ["HOME_TEAM", "DATE"],
        how = "left"
    )
    df_away_home_merge = df_new_partial_2.merge(
        right = df_new_partial_1,
        left_on = "AWAY_TEAM",
        right_on = "HOME_TEAM",
        how = "inner"
    )
    df_away_home_merge["DATE_DIFF"] = (df_away_home_merge["DATE_x"] - df_away_home_merge["DATE_y"]).dt.days
    df_away_home_merge_1 = df_away_home_merge[[
        "HOME_TEAM_x",
        "AWAY_TEAM_x",
        "DATE_x",
        "DATE_y",
        "HOME_TEAM_LAST_5_HOME_WINS",
        "HOME_TEAM_LAST_5_HOME_DRAWS",
        "HOME_TEAM_LAST_5_HOME_LOSSES",
        "DATE_DIFF"
    ]]
    df_away_home_merge_1 = df_away_home_merge_1[df_away_home_merge_1["DATE_DIFF"] >= 0]
    df_away_home_merge_1["ROW_NUM"] = df_away_home_merge_1.sort_values("DATE_DIFF", ascending = True).groupby(["AWAY_TEAM_x", "DATE_x"]).cumcount() + 1
    df_away_home_merge_1_final = df_away_home_merge_1[df_away_home_merge_1["ROW_NUM"] == 1].reset_index(drop = True)
    df_away_home_merge_1_final = df_away_home_merge_1_final.rename(columns = {
        "AWAY_TEAM_x": "AWAY_TEAM",
        "DATE_x": "DATE",
        "HOME_TEAM_LAST_5_HOME_WINS": "AWAY_TEAM_LAST_5_HOME_WINS",
        "HOME_TEAM_LAST_5_HOME_DRAWS": "AWAY_TEAM_LAST_5_HOME_DRAWS",
        "HOME_TEAM_LAST_5_HOME_LOSSES": "AWAY_TEAM_LAST_5_HOME_LOSSES"
    })
    df_away_home_merge_1_final = df_away_home_merge_1_final[[
        "AWAY_TEAM",
        "DATE",
        "AWAY_TEAM_LAST_5_HOME_WINS",
        "AWAY_TEAM_LAST_5_HOME_DRAWS",
        "AWAY_TEAM_LAST_5_HOME_LOSSES"
    ]]
    df_new = pd.merge(
        left = df_new, 
        right = df_away_home_merge_1_final,
        on = ["AWAY_TEAM", "DATE"],
        how = "left"
    )
    return(df_new)


# Perform advanced feature engineering (GD over last 5 matches)
def epl_feature_engineering_2():
    df = epl_feature_engineering_1()
    df_home = df[[
        "HOME_TEAM",
        "SEASON",
        "DATE",
        "DIFF_HA_FT_GOALS"
    ]]
    df_away = df[[
        "AWAY_TEAM",
        "SEASON",
        "DATE",
        "DIFF_HA_FT_GOALS"
    ]]
    df_home["HOME_AWAY"] = "HOME"
    df_away["HOME_AWAY"] = "AWAY"
    df_home = df_home.rename(columns = {
        "HOME_TEAM": "TEAM",
        "DIFF_HA_FT_GOALS": "FT_GD"
    })
    df_away = df_away.rename(columns = {
        "AWAY_TEAM": "TEAM",
        "DIFF_HA_FT_GOALS": "FT_GD"
    })
    df_away["FT_GD"] = - df_away["FT_GD"]
    df_full = pd.concat([df_home, df_away]).reset_index(drop = True)
    df_new = pd.DataFrame()
    for team in df_full["TEAM"].unique():
        df_team = df_full[df_full["TEAM"] == team]
        df_team = df_team.sort_values("DATE", ascending = True).reset_index(drop = True)
        for season in df_full["SEASON"].unique():
            df_team_season = df_team[df_team["SEASON"] == season]
            df_team_season.sort_values("DATE", ascending = True).reset_index(drop = True)
            df_team_season["GD_LAST_5"] = df_team_season["FT_GD"].rolling(5).sum()
            df_team_season["GD_LAST_5"] = df_team_season["GD_LAST_5"].shift(1)
            df_team_season["WINS_LAST_5"] = df_team_season["FT_GD"].apply(
                lambda x: 1 if x > 0 else 0
            ).rolling(5).sum()
            df_team_season["WINS_LAST_5"] = df_team_season["WINS_LAST_5"].shift(1)
            df_team_season["DRAWS_LAST_5"] = df_team_season["FT_GD"].apply(
                lambda x: 1 if x == 0 else 0
            ).rolling(5).sum()
            df_team_season["DRAWS_LAST_5"] = df_team_season["DRAWS_LAST_5"].shift(1)
            df_team_season["LOSSES_LAST_5"] = df_team_season["FT_GD"].apply(
                lambda x: 1 if x < 0 else 0
            ).rolling(5).sum()
            df_team_season["LOSSES_LAST_5"] = df_team_season["LOSSES_LAST_5"].shift(1)
            df_team_season["GAME_DAY_NUM"] = df_team_season.groupby("TEAM").cumcount() + 1
            df_team_season["WINS_THIS_SEASON"] = df_team_season["FT_GD"].apply(
                lambda x: 1 if x > 0 else 0
            ).cumsum()
            df_team_season["WINS_THIS_SEASON"] = df_team_season["WINS_THIS_SEASON"].shift(1)
            df_team_season["DRAWS_THIS_SEASON"] = df_team_season["FT_GD"].apply(
                lambda x: 1 if x == 0 else 0
            ).cumsum()
            df_team_season["DRAWS_THIS_SEASON"] = df_team_season["DRAWS_THIS_SEASON"].shift(1)
            df_team_season["LOSSES_THIS_SEASON"] = df_team_season["FT_GD"].apply(
                lambda x: 1 if x < 0 else 0
            ).cumsum()
            df_team_season["LOSSES_THIS_SEASON"] = df_team_season["LOSSES_THIS_SEASON"].shift(1)
            df_new = pd.concat([df_new, df_team_season])
    df_new = df_new.reset_index(drop = True)
    df_new_1 = df_new[[
        "TEAM",
        "DATE",
        "GD_LAST_5",
        "WINS_LAST_5",
        "DRAWS_LAST_5",
        "LOSSES_LAST_5",
        "GAME_DAY_NUM",
        "WINS_THIS_SEASON",
        "DRAWS_THIS_SEASON",
        "LOSSES_THIS_SEASON"
    ]]
    df_new_home = df_new_1.rename(columns = {
        "TEAM": "HOME_TEAM",
        "GD_LAST_5": "HOME_TEAM_GD_LAST_5",
        "WINS_LAST_5": "HOME_TEAM_WINS_LAST_5",
        "DRAWS_LAST_5": "HOME_TEAM_DRAWS_LAST_5",
        "LOSSES_LAST_5": "HOME_TEAM_LOSSES_LAST_5",
        "GAME_DAY_NUM": "HOME_TEAM_GAME_NUM",
        "WINS_THIS_SEASON": "HOME_TEAM_WINS_THIS_SEASON",
        "DRAWS_THIS_SEASON": "HOME_TEAM_DRAWS_THIS_SEASON",
        "LOSSES_THIS_SEASON": "HOME_TEAM_LOSSES_THIS_SEASON"
    })
    df_new_away = df_new_1.rename(columns = {
        "TEAM": "AWAY_TEAM",
        "GD_LAST_5": "AWAY_TEAM_GD_LAST_5",
        "WINS_LAST_5": "AWAY_TEAM_WINS_LAST_5",
        "DRAWS_LAST_5": "AWAY_TEAM_DRAWS_LAST_5",
        "LOSSES_LAST_5": "AWAY_TEAM_LOSSES_LAST_5",
        "GAME_DAY_NUM": "AWAY_TEAM_GAME_NUM",
        "WINS_THIS_SEASON": "AWAY_TEAM_WINS_THIS_SEASON",
        "DRAWS_THIS_SEASON": "AWAY_TEAM_DRAWS_THIS_SEASON",
        "LOSSES_THIS_SEASON": "AWAY_TEAM_LOSSES_THIS_SEASON"
    })
    df = pd.merge(
        left = df,
        right = df_new_home,
        on = ["HOME_TEAM", "DATE"],
        how = "inner"
    )
    df = pd.merge(
        left = df,
        right = df_new_away,
        on = ["AWAY_TEAM", "DATE"],
        how = "inner"
    )
    df["HOME_TEAM_PTS"] = 3 * df["HOME_TEAM_WINS_THIS_SEASON"] + df["HOME_TEAM_DRAWS_THIS_SEASON"]
    df["AWAY_TEAM_PTS"] = 3 * df["AWAY_TEAM_WINS_THIS_SEASON"] + df["AWAY_TEAM_DRAWS_THIS_SEASON"]
    df["DIFF_HA_PTS"] = df["HOME_TEAM_PTS"] - df["AWAY_TEAM_PTS"]
    return(df)


# Perform advanced feature engineering:
# Pos. in table, month of season, is the team in the 'big 6', and was the team promoted from the EFL Championship last season?
def epl_feature_engineering_3():
    df = epl_feature_engineering_2()
    df_home = df[[
        "HOME_TEAM",
        "SEASON",
        "HOME_TEAM_GAME_NUM",
        "HOME_TEAM_PTS"
    ]]
    df_away = df[[
        "AWAY_TEAM",
        "SEASON",
        "AWAY_TEAM_GAME_NUM",
        "AWAY_TEAM_PTS"
    ]]
    df_home["HOME_AWAY"] = "HOME"
    df_away["HOME_AWAY"] = "AWAY"
    df_home = df_home.rename(columns = {
        "HOME_TEAM": "TEAM",
        "HOME_TEAM_GAME_NUM": "GAME_NUM",
        "HOME_TEAM_PTS": "PTS"
    })
    df_away = df_away.rename(columns = {
        "AWAY_TEAM": "TEAM",
        "AWAY_TEAM_GAME_NUM": "GAME_NUM",
        "AWAY_TEAM_PTS": "PTS"
    })
    df_full = pd.concat([df_home, df_away]).reset_index(drop = True)
    df_new = pd.DataFrame()
    for season in df_full["SEASON"].unique():
        df_season = df_full[df_full["SEASON"] == season]
        df_season = df_season.sort_values(["GAME_NUM"], ascending = True).reset_index(drop = True)
        for game_num in df_season["GAME_NUM"].unique():
            df_season_game = df_season[df_season["GAME_NUM"] == game_num].sort_values("PTS", ascending = True).reset_index(drop = True)
            df_season_game["POSITION"] = df_season_game.sort_values("PTS", ascending = False).groupby(["SEASON", "GAME_NUM"]).cumcount() + 1
            df_new = pd.concat([df_new, df_season_game]).reset_index(drop = True)
    df_new = df_new.reset_index(drop = True)
    df_new_1 = df_new[[
        "TEAM",
        "SEASON",
        "GAME_NUM",
        "POSITION"
    ]]
    df_new_home = df_new_1.rename(columns = {
        "TEAM": "HOME_TEAM",
        "GAME_NUM": "HOME_TEAM_GAME_NUM",
        "POSITION": "HOME_TEAM_POSITION"
    })
    df_new_away = df_new_1.rename(columns = {
        "TEAM": "AWAY_TEAM",
        "GAME_NUM": "AWAY_TEAM_GAME_NUM",
        "POSITION": "AWAY_TEAM_POSITION"
    })
    df = pd.merge(
        left = df,
        right = df_new_home,
        on = ["HOME_TEAM", "HOME_TEAM_GAME_NUM", "SEASON"],
        how = "inner"
    )
    df = pd.merge(
        left = df,
        right = df_new_away,
        on = ["AWAY_TEAM", "AWAY_TEAM_GAME_NUM", "SEASON"],
        how = "inner"
    )
    df["HOME_TEAM_BIG_6_FLAG"] = df["HOME_TEAM"].apply(
        lambda x: 1 if x in [
            "Arsenal", 
            "Chelsea",
            "Liverpool",
            "Man City",
            "Man United",
            "Tottenham"
        ] else 0
    )
    df["AWAY_TEAM_BIG_6_FLAG"] = df["AWAY_TEAM"].apply(
        lambda x: 1 if x in [
            "Arsenal", 
            "Chelsea",
            "Liverpool",
            "Man City",
            "Man United",
            "Tottenham"
        ] else 0
    )
    df_prom = pd.DataFrame()
    for i in df["HOME_TEAM"].unique():
        df_team = df[df["HOME_TEAM"] == i]
        for j in pd.to_numeric(df["SEASON"].str[2:4].unique()):
            df_team_season = df_team[df_team["SEASON"] == f"20{j}/20{j+1}"]
            if (i in df[df["SEASON"] == f"20{j}/20{j+1}"]["HOME_TEAM"].unique()) and (i in df[df["SEASON"] == f"20{j-1}/20{j}"]["HOME_TEAM"].unique()):
                df_team_season["HOME_TEAM_PROMOTED_FROM_LAST_SEASON_FLAG"] = 0
            else:
                df_team_season["HOME_TEAM_PROMOTED_FROM_LAST_SEASON_FLAG"] = 1
            df_prom = pd.concat([df_prom, df_team_season])
    df_prom = df_prom.reset_index(drop = True)
    df_team_season_prom = df_prom[[
        "HOME_TEAM",
        "SEASON",
        "HOME_TEAM_PROMOTED_FROM_LAST_SEASON_FLAG"
    ]].drop_duplicates().reset_index(drop = True)
    df_team_season_prom = df_team_season_prom.rename(columns = {
        "HOME_TEAM": "AWAY_TEAM",
        "HOME_TEAM_PROMOTED_FROM_LAST_SEASON_FLAG": "AWAY_TEAM_PROMOTED_FROM_LAST_SEASON_FLAG"
    })
    df_final = pd.merge(
        left = df_prom,
        right = df_team_season_prom,
        on = ["AWAY_TEAM", "SEASON"],
        how = "inner"
    )
    pl_month_mapping = {
        "07": 0,
        "08": 1,
        "09": 2,
        "10": 3,
        "11": 4,
        "12": 5,
        "01": 6,
        "02": 7,
        "03": 8,
        "04": 9,
        "05": 10,
        "06": 11
    }
    df_final["SEASON_MONTH_NUM"] = [
        pl_month_mapping[i] for i in df_final["DATE"].astype(str).str[5:7]
    ]
    return(df_final)


# Perform advanced feature engineering (shots/shots on target last 5 games)
def epl_feature_engineering_4():
    df = epl_feature_engineering_3()
    df_home = df[[
        "HOME_TEAM",
        "SEASON",
        "DATE",
        "HS",
        "HST"
    ]]
    df_away = df[[
        "AWAY_TEAM",
        "SEASON",
        "DATE",
        "AS",
        "AST"
    ]]
    df_home["HOME_AWAY"] = "HOME"
    df_away["HOME_AWAY"] = "AWAY"
    df_home = df_home.rename(columns = {
        "HOME_TEAM": "TEAM",
        "HS": "SHOTS",
        "HST": "SHOTS_ON_TARGET"
    })
    df_away = df_away.rename(columns = {
        "AWAY_TEAM": "TEAM",
        "AS": "SHOTS",
        "AST": "SHOTS_ON_TARGET"
    })
    df_full = pd.concat([df_home, df_away]).reset_index(drop = True)
    df_new = pd.DataFrame()
    for team in df_full["TEAM"].unique():
        df_team = df_full[df_full["TEAM"] == team]
        df_team = df_team.sort_values("DATE", ascending = True).reset_index(drop = True)
        for season in df_team["SEASON"].unique():
            df_team_season = df_team[df_team["SEASON"] == season]
            df_team_season = df_team_season.sort_values("DATE", ascending = True).reset_index(drop = True)
            df_team_season["SHOTS_LAST_5"] = df_team_season["SHOTS"].rolling(5).sum()
            df_team_season["SHOTS_LAST_5"] = df_team_season["SHOTS_LAST_5"].shift(1)
            df_team_season["SHOTS_ON_TARGET_LAST_5"] = df_team_season["SHOTS_ON_TARGET"].rolling(5).sum()
            df_team_season["SHOTS_ON_TARGET_LAST_5"] = df_team_season["SHOTS_ON_TARGET_LAST_5"].shift(1)
            df_new = pd.concat([df_new, df_team_season])
    df_new_1 = df_new[[
        "TEAM",
        "DATE",
        "SHOTS_LAST_5",
        "SHOTS_ON_TARGET_LAST_5"
    ]].reset_index(drop = True)
    df_new_home = df_new_1.rename(columns = {
        "TEAM": "HOME_TEAM",
        "SHOTS_LAST_5": "HOME_TEAM_SHOTS_LAST_5",
        "SHOTS_ON_TARGET_LAST_5": "HOME_TEAM_SHOTS_ON_TARGET_LAST_5"
    })
    df_new_away = df_new_1.rename(columns = {
        "TEAM": "AWAY_TEAM",
        "SHOTS_LAST_5": "AWAY_TEAM_SHOTS_LAST_5",
        "SHOTS_ON_TARGET_LAST_5": "AWAY_TEAM_SHOTS_ON_TARGET_LAST_5"
    })
    df = pd.merge(
        left = df,
        right = df_new_home,
        on = ["HOME_TEAM", "DATE"],
        how = "inner"
    )
    df = pd.merge(
        left = df,
        right = df_new_away,
        on = ["AWAY_TEAM", "DATE"],
        how = "inner"
    )
    df["DIFF_HA_SHOTS_LAST_5"] = df["HOME_TEAM_SHOTS_LAST_5"] - df["AWAY_TEAM_SHOTS_LAST_5"]
    df["DIFF_HA_SHOTS_ON_TARGET_LAST_5"] = df["HOME_TEAM_SHOTS_ON_TARGET_LAST_5"] - df["AWAY_TEAM_SHOTS_ON_TARGET_LAST_5"]
    return(df)


# Clean up of data post feature engineering
def clean_augmented_epl_data():
    df = epl_feature_engineering_4()
    df["DIFF_HA_LAST_5_HOME_WINS"] = df["HOME_TEAM_LAST_5_HOME_WINS"] - df["AWAY_TEAM_LAST_5_HOME_WINS"]
    df["DIFF_HA_LAST_5_HOME_DRAWS"] = df["HOME_TEAM_LAST_5_HOME_DRAWS"] - df["AWAY_TEAM_LAST_5_HOME_DRAWS"]
    df["DIFF_HA_LAST_5_HOME_LOSSES"] = df["HOME_TEAM_LAST_5_HOME_LOSSES"] - df["AWAY_TEAM_LAST_5_HOME_LOSSES"]
    df["DIFF_HA_LAST_5_AWAY_WINS"] = df["HOME_TEAM_LAST_5_AWAY_WINS"] - df["AWAY_TEAM_LAST_5_AWAY_WINS"]
    df["DIFF_HA_LAST_5_AWAY_DRAWS"] = df["HOME_TEAM_LAST_5_AWAY_DRAWS"] - df["AWAY_TEAM_LAST_5_AWAY_DRAWS"]
    df["DIFF_HA_LAST_5_AWAY_LOSSES"] = df["HOME_TEAM_LAST_5_AWAY_LOSSES"] - df["AWAY_TEAM_LAST_5_AWAY_LOSSES"]
    df["DIFF_HA_LAST_5_WINS"] = df["HOME_TEAM_WINS_LAST_5"] - df["AWAY_TEAM_WINS_LAST_5"]
    df["DIFF_HA_LAST_5_DRAWS"] = df["HOME_TEAM_DRAWS_LAST_5"] - df["AWAY_TEAM_DRAWS_LAST_5"]
    df["DIFF_HA_LAST_5_LOSSES"] = df["HOME_TEAM_LOSSES_LAST_5"] - df["AWAY_TEAM_LOSSES_LAST_5"]
    df["DIFF_HA_GD_LAST_5"] = df["HOME_TEAM_GD_LAST_5"] - df["AWAY_TEAM_GD_LAST_5"]
    df["DIFF_HA_PTS_THIS_SEASON"] = df["HOME_TEAM_PTS"] - df["AWAY_TEAM_PTS"]
    df["DIFF_HA_CURRENT_POSITION"] = df["HOME_TEAM_POSITION"] - df["AWAY_TEAM_POSITION"]
    df["TOT_GOALS"] = df["FTHG"] + df["FTAG"]
    df_final = df[[
        "HOME_TEAM",
        "AWAY_TEAM",
        "SEASON",
        "DATE",
        "REFEREE",
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
        "AWAY_TEAM_GAME_NUM",
        "RESULT",
        "FTHG",
        "FTAG",
        "DIFF_HA_FT_GOALS",
        "TOT_GOALS"
    ]]
    df_out = df_final[
        (df_final["HOME_TEAM_GAME_NUM"] >= 12) &
        (df_final["AWAY_TEAM_GAME_NUM"] >= 12) &
        (df_final["SEASON"] > min(df_final["SEASON"].unique()))
    ].reset_index(drop = True)
    return(df_out)


# Push raw data to PostgreSQL database
def push_preproccessed_historical_data_to_db(connection_params, schema_name, table_name):
    conn = psycopg2.connect(
        host = connection_params["host"],
        database = connection_params["database"],
        user = connection_params["user"],
        password = connection_params["password"]
    )
    engine = create_engine('postgresql://' + connection_params["user"] + ':' + connection_params["password"] + '@' + connection_params["host"] + ':5432/' + connection_params["database"])
    df = clean_augmented_epl_data()
    conn.autocommit = True
    cursor = conn.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS {schema_name}.{table_name}")
    cursor.execute(
        f'''
            CREATE TABLE {schema_name}.{table_name} (
                HOME_TEAM VARCHAR(100),
                AWAY_TEAM VARCHAR(100),
                "SEASON" VARCHAR(100),
                DATE DATE,
                REFEREE VARCHAR(100),
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
                RESULT VARCHAR(100),
                FTHG INT,
                FTAG INT,
                DIFF_HA_FT_GOALS INT,
                TOT_GOALS INT
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


# Execute the pipeline
def run_etl_pipeline():
    connection_params = json.load(
        open("/Users/callanroff/Desktop/Acc. Keyzzz/postgresql_conn_params.json", "r")
    )
    push_preproccessed_historical_data_to_db(
        connection_params = connection_params, 
        schema_name="web_scraping", 
        table_name="epl_data"
    )

