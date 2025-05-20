from pathlib import Path
import os
import pandas as pd
import catboost as cb
import pytz
import time
import json
from utils import create_features_and_target
import os


# Define Paths

TRAINED_MODEL_DIR = Path('/app/ingested_program/last_model.cbm')

TEST_DATA_DIR = os.path.join('/app','data', 'dataset', 'test')
TEST_PREDS_FP = Path('/app/output/prediction.json')

initial_states_file = os.path.join('/app/input_data',"initial_states.csv")
omni2_path = os.path.join(TEST_DATA_DIR, "omni2")

# Load models for fill na of Altitude, Logitude and Latitude

model_alt = cb.CatBoostRegressor()
model_alt.load_model('/app/ingested_program/Altitude_Model.cbm')

model_lat = cb.CatBoostRegressor()
model_lat.load_model('/app/ingested_program/Latitude_Model.cbm')

model_long = cb.CatBoostRegressor()
model_long.load_model('/app/ingested_program/Longitude_Model.cbm')

# Load initial states

initial_states = pd.read_csv(initial_states_file)
initial_states = pd.read_csv(initial_states_file, usecols=['File ID', 'Timestamp', 'Semi-major Axis (km)',
                             'Eccentricity', 'Inclination (deg)', 'RAAN (deg)', 'Argument of Perigee (deg)', 'True Anomaly (deg)',
                             'Latitude (deg)', 'Longitude (deg)', 'Altitude (km)'])
initial_states['Timestamp'] = pd.to_datetime(initial_states['Timestamp'])
initial_states = initial_states.set_index('File ID')

initial_states['DOY'] = initial_states['Timestamp'].dt.day_of_year
initial_states['HOUR'] = initial_states['Timestamp'].dt.hour
initial_states['Year'] = initial_states['Timestamp'].dt.year
initial_states['bartels_phase'] = (initial_states['DOY'] % 27)/27 * 360

# Correct Initial states files

initial_states['Altitude (km)'] = initial_states['Altitude (km)']/1000

initial_states['Altitude (km)'] = initial_states['Altitude (km)'].mask(initial_states['Altitude (km)']>1_000_000_000)
initial_states['Latitude (deg)'] = initial_states['Latitude (deg)'].mask(initial_states['Latitude (deg)']>1_000_000)
initial_states['Longitude (deg)'] = initial_states['Longitude (deg)'].mask(initial_states['Longitude (deg)']>1_000_000)

col_to_pred = ['Semi-major Axis (km)',
               'Eccentricity',
               'Inclination (deg)',
               'RAAN (deg)',
               'Argument of Perigee (deg)',
               'True Anomaly (deg)']

altitude_estimated = pd.Series(model_alt.predict(initial_states[col_to_pred]),index = initial_states.index)
latitude_estimated = pd.Series(model_lat.predict(initial_states[col_to_pred]),index = initial_states.index)
longitude_estimated = pd.Series(model_long.predict(initial_states[col_to_pred]),index = initial_states.index)

initial_states['Altitude (km)'] = initial_states['Altitude (km)'].fillna(altitude_estimated)
initial_states['Latitude (deg)'] = initial_states['Latitude (deg)'].fillna(latitude_estimated)
initial_states['Longitude (deg)'] = initial_states['Longitude (deg)'].fillna(longitude_estimated)

# Process each row of the initial states

model_task_1 = cb.CatBoostRegressor()
model_task_1.load_model(TRAINED_MODEL_DIR)

predictions = {}
for file_id in initial_states.index:
    # Load corresponding OMNI2 data
    omni2_file = os.path.join(omni2_path, f"omni2-{file_id:05}.csv")

    if not os.path.exists(omni2_file):
        print(f"OMNI2 file {omni2_file} not found! Skipping...")
        continue

    df_features = create_features_and_target(id = file_id,
                                             initial_states = initial_states,
                                             omni_path = omni2_file, 
                                             predict_mean=False,
                                             training_mode=False).reset_index()

    df_features = df_features[model_task_1.feature_names_ +['Timestamp'] ]
    result = pd.Series(model_task_1.predict(df_features.drop(['Timestamp'],axis=1))/ 10**12)

    max_per_altitude = df_features['Altitude (km)'].iloc[0] * -2.6363599e-14 + 1.8172370431533262e-11

    print((result>max_per_altitude).sum())
   
    result = result.mask(result<2.418298862833011e-14,2.418298862833011e-14) 
    result = result.mask(result>max_per_altitude,max_per_altitude) 

    predictions[file_id] = {
        "Timestamp": df_features["Timestamp"].dt.tz_localize(pytz.utc).apply(lambda win:win.isoformat()).to_list(),
        "Orbit Mean Density (kg/m^3)": result.to_list()
    }
    print(f"Model execution for {file_id} Finished")


with open(TEST_PREDS_FP, "w") as outfile:
    json.dump(predictions, outfile)

print("Saved predictions to: {}".format(TEST_PREDS_FP))
