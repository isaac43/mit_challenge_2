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


def create_features_and_target(id,df_train,omni = None,sat = None,path='phase_1/',predict_mean=True):

    if omni is None:
        all_files_omni = os.listdir(path+"omni2/")
        all_files_dens = os.listdir(path+ "sat_density/")
        file_omni = [i for i in all_files_omni if i.split('-')[1]==id][0]
        file_sat = [i for i in all_files_dens if i.split('-')[1]==id][0]
        data_omni = pd.read_csv(path+'omni2/'+file_omni)
        data_sat = pd.read_csv(path+'sat_density/'+file_sat)
    else:
        data_omni = omni[id]
        data_sat = sat[id]

    data_omni['Timestamp'] = pd.to_datetime(data_omni['Timestamp'])
    data_omni = data_omni.ffill()
    data_sat['Timestamp'] = pd.to_datetime(data_sat['Timestamp'],format='%Y-%m-%d %H:%M:%S')

    df_features = pd.concat([data_omni.drop('Timestamp',axis=1).mean(0),data_omni.iloc[-1]],axis=1)
    df_features.columns = ['mean','last']
    df_features = df_features.unstack()
    df_features.index = df_features.index.get_level_values(0)+'_'+ df_features.index.get_level_values(1)
    df_features['last_Timestamp'] = pd.to_datetime(df_features['last_Timestamp'],format='%Y-%m-%d %H:%M:%S')
    df_position = df_train.loc[id]
    df_features = pd.concat([df_position,df_features],axis=0)
    if not predict_mean :
        df_features = pd.concat([df_features.to_frame().T,data_sat]).ffill().dropna(subset='Orbit Mean Density (kg/m^3)')
        df_features['Time_until_y'] = (df_features['Timestamp'] - df_features['last_Timestamp']).dt.seconds.copy()
    else:
        df_features.loc['Target'] = data_sat['Orbit Mean Density (kg/m^3)'].mean()

    return df_features