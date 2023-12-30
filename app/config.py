# Import required libraries 
import os
from pathlib import Path


# Specify path defaults
APP_PATH = Path(os.environ["PYTHONPATH"])
INPUT_PATH = APP_PATH / "input/"
OUTPUT_PATH = APP_PATH / "output/"
MODEL_PATH = APP_PATH / "app/model"


# Email
SMTP_SERVER = "smtp.gmail.com"
PORT = 465


# Default model inputs (features)
X_COLS = [
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