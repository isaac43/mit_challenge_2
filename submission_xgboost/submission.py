from pathlib import Path
import os
import pandas as pd
import catboost as cb
import pytz
import time
import json
from utils import create_features_and_target
import os

TRAINED_MODEL_DIR = Path('/app/ingested_program/model_by_timestamp.cbm')

TEST_DATA_DIR = os.path.join('/app','data', 'dataset', 'test')
TEST_PREDS_FP = Path('/app/output/prediction.json')

# Paths

initial_states_file = os.path.join(TEST_DATA_DIR, "initial_states.csv")
omni2_path = os.path.join(TEST_DATA_DIR, "omni2")

initial_states = pd.read_csv(initial_states_file)

# Load initial states
initial_states = pd.read_csv(initial_states_file, usecols=['File ID', 'Timestamp', 'Semi-major Axis (km)',
                             'Eccentricity', 'Inclination (deg)', 'RAAN (deg)', 'Argument of Perigee (deg)', 'True Anomaly (deg)',
                             'Latitude (deg)', 'Longitude (deg)', 'Altitude (km)'])
initial_states['Timestamp'] = pd.to_datetime(initial_states['Timestamp'])
initial_states = initial_states.set_index('File ID')

# Process each row of the initial states

model = cb.CatBoostRegressor()
model.load_model(TRAINED_MODEL_DIR)
predictions = {}
for file_id in initial_states.index:
    # Load corresponding OMNI2 data
    omni2_file = os.path.join(omni2_path, f"omni2-{file_id:05}.csv")

    if not os.path.exists(omni2_file):
        print(f"OMNI2 file {omni2_file} not found! Skipping...")
        continue

    df_features = create_features_and_target(id = file_id,
                                             initial_states = initial_states,
                                             #omni = None,
                                             omni_path = omni2_file, 
                                             predict_mean=False,
                                             training_mode=False).reset_index()
    df_features = df_features[model.feature_names_ +['Timestamp'] ]

    result = model.predict(df_features.drop(['Timestamp'],axis=1))/ 10**12
    predictions[file_id] = {
        "Timestamp": df_features["Timestamp"].dt.tz_localize(pytz.utc).apply(lambda win:win.isoformat()).to_list(),
        "Orbit Mean Density (kg/m^3)": list(result)
    }
    print(f"Model execution for {file_id} Finished")


with open(TEST_PREDS_FP, "w") as outfile:
    json.dump(predictions, outfile)

print("Saved predictions to: {}".format(TEST_PREDS_FP))
