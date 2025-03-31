import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from datetime import datetime, timezone
import os
from scipy.stats import linregress, entropy
from scipy.signal import find_peaks, periodogram
from statsmodels.tsa.stattools import adfuller, acf
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
        data_omni = omni#[id]
    
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
        data_sat = sat#[id]
    data_sat['Timestamp'] = pd.to_datetime(data_sat['Timestamp'],format='%Y-%m-%d %H:%M:%S')
    data_sat['File Id'] = id

    # df_features = pd.concat([data_omni.drop(['Timestamp'],axis=1).mean(0).add_prefix('mean_'),
    #                          data_omni.drop(['Timestamp'],axis=1).max(0).add_prefix('max_'),
    #                          data_omni.drop(['Timestamp'],axis=1).min(0).add_prefix('min_'),
    #                          data_omni.drop(['Timestamp'],axis=1).iloc[-1].add_prefix('last_')],axis=0)
    
    df_features = extract_features(data_omni.drop(['Timestamp'],axis=1))
    df_features = df_features.unstack().reset_index()
    df_features.index = df_features['level_0'] + '_' + df_features['level_1']
    df_features = df_features[[0]].T
    
    df_features['File Id'] = id    
    df_position = initial_states.loc[[id]].rename({'Timestamp':'Timestamp_initial'},axis=1)
    df_position['File Id'] = id
    df_features = pd.merge(df_features,df_position,on='File Id')

    # df_features['Log_lat'] = np.log(df_features['Latitude (deg)'])  
    # df_features['Exp_lat'] = np.exp(df_features['Latitude (deg)'])  
    if training_mode:
        if not predict_mean :
            df_features = pd.merge(data_sat,df_features,on='File Id')
            df_features['Time_until_y'] = (df_features['Timestamp'] - df_features['Timestamp_initial']).dt.seconds
            df_features = df_features.set_index('Timestamp').drop('File Id',axis=1)
        else:
            df_features['Target'] = data_sat['Orbit Mean Density (kg/m^3)'].mean()
            # print(df_features)
            # df_features = df_features.to_frame().T.set_index('File Id')
    else:
        if not predict_mean:
            df_features = pd.merge(data_sat,df_features,on='File Id')
            df_features['Time_until_y'] = (df_features['Timestamp'] - df_features['Timestamp_initial']).dt.seconds
            df_features = df_features.set_index('Timestamp').drop('File Id',axis=1)
        else:
            df_features = df_features.to_frame().T.set_index('File Id')

    return df_features
from scipy import stats


def extract_features(df):
    """
    Extract statistical features from a DataFrame for each column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with numerical columns
    
    Returns:
    pd.DataFrame: DataFrame with extracted features as rows and original columns as columns
    """
    feature_definitions = [
        ('min', 'min'),
        ('max', 'max'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('var', 'var'),
        ('last', lambda x: x.iloc[-1]),
        ('range', lambda x: x.max() - x.min()),
        ('sum', 'sum'),
        ('skew', 'skew'),
        ('kurtosis', 'kurtosis'),
        ('25th_percentile', lambda x: x.quantile(0.25)),
        ('75th_percentile', lambda x: x.quantile(0.75)),
        ('IQR', lambda x: x.quantile(0.75) - x.quantile(0.25)),
        ('mad', lambda x: (x - x.median()).abs().median()),  # Mean Absolute Deviation

        # Trend/Drift Features
        ('trend_slope', lambda x: linregress(np.arange(len(x)), x).slope),
        ('trend_pvalue', lambda x: linregress(np.arange(len(x)), x).pvalue),
        
        # Temporal Change Features
        ('mean_abs_change', lambda x: np.mean(np.abs(np.diff(x)))),
        ('max_abs_change', lambda x: np.max(np.abs(np.diff(x)))),
        ('mean_second_derivative', lambda x: np.mean(np.diff(x, n=2))),
        
        # # Temporal Correlation Features
        ('autocorr_lag1', lambda x: x.autocorr(lag=1)),
        ('autocorr_lag5', lambda x: x.autocorr(lag=5)),
        # ('partial_autocorr', lambda x: acf(x, nlags=5, fft=True)[-1]),
        
        # Temporal Event Features
        ('peak_count', lambda x: len(find_peaks(x)[0])),
        ('trough_count', lambda x: len(find_peaks(-x)[0])),
        ('zero_crossings', lambda x: len(np.where(np.diff(np.sign(x - x.mean())))[0])),
        
        # Windowed Statistics
        ('max_rolling_mean_7', lambda x: x.rolling(7).mean().max()),
        ('min_rolling_std_7', lambda x: x.rolling(7).std().min()),
    ]
    
    # Extract functions from feature definitions
    functions = [func for _, func in feature_definitions]
    
    # Calculate features
    features_df = df.agg(functions)
    
    # Set proper feature names
    features_df.index = [name for name, _ in feature_definitions]
    
    return features_df



def extract_temporal_features(df):
    """
    Extract temporal features from a time-ordered DataFrame.
    Returns features where rows represent statistics and columns match original data.
    """
    # Helper functions for temporal features
    def _hurst_exponent(series):
        lags = range(2, 100)
        tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
        return np.polyfit(np.log(lags), np.log(tau), 1)[0]*2.0

    def _spectral_entropy(series):
        print(series)
        psd = periodogram(series)[1]
        psd_norm = psd / psd.sum()
        return -np.sum(psd_norm * np.log(psd_norm + 1e-12))

    temporal_feature_definitions = [
        # Trend/Drift Features
        ('trend_slope', lambda x: linregress(np.arange(len(x)), x).slope),
        ('trend_pvalue', lambda x: linregress(np.arange(len(x)), x).pvalue),
        
        # Temporal Change Features
        ('mean_abs_change', lambda x: np.mean(np.abs(np.diff(x)))),
        ('max_abs_change', lambda x: np.max(np.abs(np.diff(x)))),
        ('mean_second_derivative', lambda x: np.mean(np.diff(x, n=2))),
        
        # # Temporal Correlation Features
        ('autocorr_lag1', lambda x: x.autocorr(lag=1)),
        ('autocorr_lag5', lambda x: x.autocorr(lag=5)),
        # ('partial_autocorr', lambda x: acf(x, nlags=5, fft=True)[-1]),
        
        # Temporal Event Features
        ('peak_count', lambda x: len(find_peaks(x)[0])),
        ('trough_count', lambda x: len(find_peaks(-x)[0])),
        ('zero_crossings', lambda x: len(np.where(np.diff(np.sign(x - x.mean())))[0])),
        
        # Windowed Statistics
        ('max_rolling_mean_7', lambda x: x.rolling(7).mean().max()),
        ('min_rolling_std_7', lambda x: x.rolling(7).std().min()),

        # # Frequency Domain Features
        # ('spectral_entropy', _spectral_entropy),
        #('dominant_frequency', lambda x: periodogram(x)[0][np.argmax(periodogram(x)[1])]),
        
        # # Temporal Complexity Features
        # ('hurst_exponent', _hurst_exponent),
        # ('approximate_entropy', lambda x: entropy(np.diff(x))),
        
        
        # # Stationarity Features
        # ('adf_pvalue', lambda x: adfuller(x)[1])
    ]

    # Extract functions from feature definitions
    functions = [func for _, func in temporal_feature_definitions]
    
    # Calculate features
    features_df = df.agg(functions)
    
    # Set proper feature names
    features_df.index = [name for name, _ in temporal_feature_definitions]

    return features_df