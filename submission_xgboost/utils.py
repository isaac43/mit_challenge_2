import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from datetime import datetime, timezone
import os
# noqa

def create_pred_timestamps(tm):
    new_tm = tm + timedelta(seconds=10-tm.second%10)
    date_range = pd.date_range(start=new_tm,periods=432,freq='10min')
    return date_range

def create_features_and_target(id,initial_states,omni = None,sat = None,path='phase_1/',predict_mean=True,training_mode=False,omni_path=None):

    if omni is None and omni_path is None:
        all_files_omni = os.listdir(path+"omni2/")
        file_omni = [i for i in all_files_omni if i.split('-')[1]==id][0]
        data_omni = pd.read_csv(path+'omni2/'+file_omni)
    elif omni_path is not None:
        data_omni = pd.read_csv(omni_path)
    else:
        data_omni = omni[id]
    
    data_omni['Timestamp'] = pd.to_datetime(data_omni['Timestamp'],format='%Y-%m-%d %H:%M:%S')
    data_omni = data_omni.ffill()
    
    if sat is None:
        if training_mode:
            all_files_sat = os.listdir(path+ "sat_density/")
            file_sat = [i for i in all_files_sat if i.split('-')[1]==id][0]
            data_sat = pd.read_csv(path+'sat_density/'+file_sat)
        else:
            data_sat = pd.DataFrame(create_pred_timestamps(initial_states['Timestamp'].loc[id]),columns=['Timestamp'])       
    else:
        data_sat = sat[id]
    data_sat['Timestamp'] = pd.to_datetime(data_sat['Timestamp'],format='%Y-%m-%d %H:%M:%S')
    data_sat['File Id'] = id

    df_features = pd.concat([data_omni.drop(['Timestamp'],axis=1).mean(0).add_prefix('mean_'),
                             data_omni.drop(['Timestamp'],axis=1).max(0).add_prefix('max_'),
                             data_omni.drop(['Timestamp'],axis=1).min(0).add_prefix('min_'),
                             data_omni.drop(['Timestamp'],axis=1).iloc[-1].add_prefix('last_')],axis=0)
    
    
    df_features['File Id'] = id    
    df_position = initial_states.loc[[id]].rename({'Timestamp':'Timestamp_initial'},axis=1)
    df_position['File Id'] = id
    df_features = pd.merge(df_features.to_frame().T,df_position,on='File Id')
    if training_mode:
        if not predict_mean :
            df_features = pd.merge(data_sat,df_features,on='File Id')
            df_features['Time_until_y'] = (df_features['Timestamp'] - df_features['Timestamp_initial']).dt.seconds
            df_features = df_features.set_index('Timestamp').drop('File Id',axis=1)
        else:
            df_features.loc['Target'] = data_sat['Orbit Mean Density (kg/m^3)'].mean()
            df_features = df_features.to_frame().T.set_index('File Id')
    else:
        if not predict_mean:
            df_features = pd.merge(data_sat,df_features,on='File Id')
            df_features['Time_until_y'] = (df_features['Timestamp'] - df_features['Timestamp_initial']).dt.seconds
            df_features = df_features.set_index('Timestamp').drop('File Id',axis=1)
        else:
            df_features = df_features.to_frame().T.set_index('File Id')

    return df_features