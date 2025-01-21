import pandas as pd
from datetime import datetime, timedelta
import os
# noqa

def create_pred_timestamps(tm):
    new_tm = tm + timedelta(seconds=10-tm.second%10)
    date_range = pd.date_range(start=new_tm,periods=432,freq='10min')
    return date_range

def create_features_for_prediction(id,initial_states,path_to_omni):
    
    data_omni = pd.read_csv(path_to_omni)
    data_omni['Timestamp'] = pd.to_datetime(data_omni['Timestamp'],format='%Y-%m-%d %H:%M:%S')
    data_omni = data_omni.ffill()

    data_sat = pd.DataFrame(create_pred_timestamps(initial_states['Timestamp'].loc[id]),columns=['Timestamp'])
    data_sat['File Id'] = id

    df_features = pd.concat([data_omni.drop('Timestamp',axis=1).mean(0),data_omni.drop('Timestamp',axis=1).iloc[-1]],axis=1)
    df_features.columns = ['mean','last']
    df_features = df_features.unstack()
    df_features.index = df_features.index.get_level_values(0)+'_'+ df_features.index.get_level_values(1)
    df_position = initial_states.loc[id].rename({'Timestamp':'Timestamp_initial'})
    
    df_features = pd.concat([df_position,df_features],axis=0)
    df_features['File Id'] = id
    
    df_features = pd.merge(df_features.to_frame().T,data_sat,on = 'File Id').ffill()
    df_features['Time_until_y'] = (df_features['Timestamp'] - df_features['Timestamp_initial']).dt.seconds.copy()
    
    return df_features