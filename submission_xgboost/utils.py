import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from datetime import datetime, timezone
import os
from scipy.stats import linregress, entropy
from scipy.signal import find_peaks, periodogram
from scipy import stats

#from statsmodels.tsa.stattools import adfuller, acf
# noqa

def create_pred_timestamps(tm):
    new_tm = tm + timedelta(seconds=10-tm.second%10)
    date_range = pd.date_range(start=new_tm,periods=432,freq='10min')
    return date_range

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

def solar_wind_coupling_features(df):
    """Features related to energy transfer from solar wind to magnetosphere"""
    df['epsilon_parameter'] = df['SW_Plasma_Speed_km_s'] * \
        (df['BZ_nT_GSM'].apply(lambda x: x if x < 0 else 0)**2) * \
        (np.sin(df['SW_Plasma_flow_long_angle']/2)**4 * 0.001)
    
    df['dynamical_pressure'] = 1.6726e-6 * df['SW_Proton_Density_N_cm3'] * \
        (df['SW_Plasma_Speed_km_s']**2)
    
    df['reconnection_rate'] = df['SW_Plasma_Speed_km_s'] * \
        df['BZ_nT_GSM'].apply(lambda x: abs(x) if x < 0 else 0)
    
    return df[['epsilon_parameter', 'dynamical_pressure', 'reconnection_rate']]

def magnetic_orientation_features(df):
    """Features related to IMF orientation and variability"""
    # Southward BZ dominance
    df['Bz_south_duration'] = df['BZ_nT_GSM'].rolling(6).apply(
        lambda x: np.sum(x < -5)/6, raw=True)
    
    # IMF clock angle
    df['clock_angle'] = np.arctan2(
        df['BY_nT_GSM'], df['BZ_nT_GSM']) * 180/np.pi
    
    # Cone angle variability
    df['cone_angle_std'] = df['Lat_Angle_of_B_GSE'].rolling(24).std()
    
    return df[['Bz_south_duration', 'clock_angle', 'cone_angle_std']]

def particle_flux_features(df):
    """Features for energetic particle populations"""
    # Radiation belt response features
    energy_bins = ['1_Mev', '2_Mev', '4_Mev', '10_Mev', '30_Mev', '60_Mev']
    name= []
    for i in range(len(energy_bins)-1):
        df[f'flux_ratio_{energy_bins[i]}_{energy_bins[i+1]}'] = \
            df[f'Proton_flux_>{energy_bins[i]}'] / \
            df[f'Proton_flux_>{energy_bins[i+1]}']
        name.append(f'flux_ratio_{energy_bins[i]}_{energy_bins[i+1]}')
    
    # SEP event detection
    df['sep_event_flag'] = df['Proton_flux_>10_Mev'].rolling(3).apply(
        lambda x: 1 if (x[-1]/x[0] > 100) else 0, raw=True)
    
    return df[['sep_event_flag']+name]

def geomagnetic_response_features(df):
    """Features linking solar wind to geomagnetic indices"""
    # Dst index prediction features
    df['dst_delay_3h'] = df['Dst_index_nT'].shift(-3)
    df['ae_auroral_power'] = df['AE_index_nT'] - df['AL_index_nT']
    
    # Substorm occurrence probability
    df['substorm_prob'] = 1/(1 + np.exp(
        -0.5*(df['SW_Plasma_Speed_km_s']/500 + 
              df['BZ_nT_GSM']/10 + 
              df['SW_Proton_Density_N_cm3']/10)))
    
    return df[['dst_delay_3h', 'ae_auroral_power', 'substorm_prob']]

def solar_rotation_features(df):
    """27-day solar rotation related features"""
    # Bartels rotation phase
    df['bartels_phase'] = (df['DOY'] % 27)/27 * 360
    
    # 27-day running percentiles
    for param in ['Scalar_B_nT', 'SW_Plasma_Speed_km_s']:
        df[f'{param}_27d_percentile'] = df[param].rolling(27*24).rank(pct=True)
    
    # Solar wind recurrence detection
    df['sw_recurrence'] = df['SW_Plasma_Speed_km_s'].rolling(27*24).corr(
        df['SW_Plasma_Speed_km_s'].shift(27*24))
    
    return df[['bartels_phase','Scalar_B_nT_27d_percentile', 'SW_Plasma_Speed_km_s_27d_percentile','sw_recurrence']]

def event_detection_features(df):
    """Automated detection of space weather events"""
    # Storm sudden commencement (SSC)
    df['ssc_detection'] = (df['Dst_index_nT'].diff() < -20) & \
        (df['SW_Plasma_Speed_km_s'] > 500)
    
    # CME sheath detection
    df['cme_sheath_flag'] = (df['SW_Proton_Density_N_cm3'] > 20) & \
        (df['SW_Plasma_Speed_km_s'] > 600) & \
        (df['sigma_n_N_cm3'] > 5)
    
    # Radiation belt enhancement
    df['rb_enhancement'] = (df['Proton_flux_>1_Mev'].pct_change(6,fill_method=None) > 1) & \
        (df['AE_index_nT'] > 500)
    
    return df[['ssc_detection', 'cme_sheath_flag', 'rb_enhancement']]

def extract_space_weather_features(df):
    """Master function combining all feature categories"""
    df = df.sort_values('Timestamp').reset_index(drop=True)
    
    feature_sets = [
        solar_wind_coupling_features,
        magnetic_orientation_features,
        particle_flux_features,
        geomagnetic_response_features,
        solar_rotation_features,
        event_detection_features
    ]
    
    for func in feature_sets:
        df = pd.concat([df, func(df.copy())], axis=1)
    
    # # Add temporal features from previous answer
    # df = pd.concat([df, extract_temporal_features(df)], axis=1)
    
    return df

def extract_features_from_omni(data_omni_init ):
    data_omni = data_omni_init.copy()


    #### Correct the data with na values
    incorrect_values = {
        'ID_for_IMF_spacecraft': 99,
        'ID_for_SW_Plasma_spacecraft': 99,
        'num_points_IMF_averages': 999,
        'num_points_Plasma_averages': 999,
        'Scalar_B_nT': 999.9,
        'Vector_B_Magnitude_nT': 999.9,
        'Lat_Angle_of_B_GSE': 999.9,
        'Long_Angle_of_B_GSE': 999.9,
        'BX_nT_GSE_GSM': 999.9,
        'BY_nT_GSE': 999.9,
        'BZ_nT_GSE': 999.9,
        'BY_nT_GSM': 999.9,
        'BZ_nT_GSM': 999.9,
        'RMS_magnitude_nT': 999.9,
        'RMS_field_vector_nT': 999.9,
        'RMS_BX_GSE_nT': 999.9,
        'RMS_BY_GSE_nT': 999.9,
        'RMS_BZ_GSE_nT': 999.9,
        'SW_Plasma_Temperature_K': 9999999.0,
        'SW_Proton_Density_N_cm3': 999.9,
        'SW_Plasma_Speed_km_s': 9999.0,
        'SW_Plasma_flow_long_angle': 999.9,
        'SW_Plasma_flow_lat_angle': 999.9,
        'Alpha_Prot_ratio': 9.999,
        'sigma_T_K': 9999999.0,
        'sigma_n_N_cm3': 999.9,
        'sigma_V_km_s': 9999.0,
        'sigma_phi_V_degrees': 999.9,
        'sigma_theta_V_degrees': 999.9,
        'sigma_ratio': 9.999,
        'Flow_pressure': 99.99,
        'E_electric_field': 999.99,
        'Plasma_Beta': 999.99,
        'Alfen_mach_number': 999.9,
        'Magnetosonic_Mach_number': 99.9,
        'Quasy_Invariant': 9.9999,
        'f10.7_index': 999.9,
        'AE_index_nT': 9999,
        'AL_index_nT': 99999,
        'AU_index_nT': 99999,
        'pc_index': 999.9,
        'Proton_flux_>1_Mev': 999999.99,
        'Proton_flux_>2_Mev': 99999.99,
        'Proton_flux_>4_Mev': 99999.99,
        'Proton_flux_>10_Mev': 99999.99,
        'Proton_flux_>30_Mev': 99999.99,
        'Proton_flux_>60_Mev': 99999.99
    }

    for factor in incorrect_values.keys():
        data_omni[factor] = data_omni[factor].mask(data_omni[factor]==incorrect_values[factor])


    #### Add some features based on physics
    data_omni_expanded = extract_space_weather_features(data_omni)
    data_omni_expanded['Range_Aurore_Activity'] = data_omni_expanded['AL_index_nT'] - data_omni_expanded['AU_index_nT']
        
    
    features_mode = {}
    for factor in ['ID_for_IMF_spacecraft', 'ID_for_SW_Plasma_spacecraft','num_points_IMF_averages', 'num_points_Plasma_averages']:
        features_mode[factor + '_mode'] = data_omni[factor].mode()[0]
        features_mode[factor + '_nb_unique'] = len(data_omni[factor].unique())


    columns_to_compute_math = [
        'Scalar_B_nT',
       'Vector_B_Magnitude_nT', 'Lat_Angle_of_B_GSE', 'Long_Angle_of_B_GSE',
       'BX_nT_GSE_GSM', 'BY_nT_GSE', 'BZ_nT_GSE', 'BY_nT_GSM', 'BZ_nT_GSM',
       'RMS_magnitude_nT', 'RMS_field_vector_nT', 'RMS_BX_GSE_nT',
       'RMS_BY_GSE_nT', 'RMS_BZ_GSE_nT', 'SW_Plasma_Temperature_K',
       'SW_Proton_Density_N_cm3', 'SW_Plasma_Speed_km_s',
       'SW_Plasma_flow_long_angle', 'SW_Plasma_flow_lat_angle',
       'Alpha_Prot_ratio', 'sigma_T_K', 'sigma_n_N_cm3', 'sigma_V_km_s',
       'sigma_phi_V_degrees', 'sigma_theta_V_degrees', 'sigma_ratio',
       'Flow_pressure', 'E_electric_field', 'Plasma_Beta', 'Alfen_mach_number',
       'Magnetosonic_Mach_number', 'Quasy_Invariant', 'Kp_index',
       'R_Sunspot_No', 'Dst_index_nT', 'ap_index_nT', 'f10.7_index',
       'AE_index_nT', 'AL_index_nT', 'AU_index_nT', 'Range_Aurore_Activity',
       'pc_index', 'Lyman_alpha',
       'Proton_flux_>1_Mev', 'Proton_flux_>2_Mev', 'Proton_flux_>4_Mev',
       'Proton_flux_>10_Mev', 'Proton_flux_>30_Mev', 'Proton_flux_>60_Mev',
       'Flux_FLAG','epsilon_parameter', 'dynamical_pressure',
       'reconnection_rate', 'Bz_south_duration', 'clock_angle',
       'cone_angle_std', 'sep_event_flag', 'flux_ratio_1_Mev_2_Mev',
       'flux_ratio_2_Mev_4_Mev', 'flux_ratio_4_Mev_10_Mev',
       'flux_ratio_10_Mev_30_Mev', 'flux_ratio_30_Mev_60_Mev', 'dst_delay_3h',
       'ae_auroral_power', 'substorm_prob', 'bartels_phase',
       'Scalar_B_nT_27d_percentile', 'SW_Plasma_Speed_km_s_27d_percentile',
       'sw_recurrence', 'ssc_detection', 'cme_sheath_flag', 'rb_enhancement'
    ]
    data_omni_expanded = data_omni_expanded[columns_to_compute_math]
    features_med = data_omni_expanded.median(0).add_suffix('_med').to_dict()
    features_mean = data_omni_expanded.mean(0).add_suffix('_mean').to_dict()
    features_max = data_omni_expanded.max(0).add_suffix('_max').to_dict()
    features_min = data_omni_expanded.min(0).add_suffix('_min').to_dict()

    features_std = data_omni_expanded.std(0).add_suffix('_std').to_dict()
    features_last = data_omni_expanded.ffill().iloc[-1].add_suffix('_last').to_dict()

    all_features = {**features_mode, **features_med,**features_mean,**features_std,**features_last,**features_max,**features_min}

    return all_features

    data_omni = extract_space_weather_features(data_omni)
    
    data_omni['Timestamp'] = pd.to_datetime(data_omni['Timestamp'],format='%Y-%m-%d %H:%M:%S')
    data_omni = data_omni.ffill()
    df_features = pd.concat([data_omni.drop(['Timestamp','DOY','YEAR','Hour'],axis=1).mean(0).add_prefix('mean_'),
                             data_omni.drop(['Timestamp','DOY','YEAR','Hour'],axis=1).max(0).add_prefix('max_'),
                             data_omni.drop(['Timestamp','DOY','YEAR','Hour'],axis=1).min(0).add_prefix('min_'),
                             data_omni.drop(['Timestamp'],axis=1).iloc[-1].add_prefix('last_')],axis=0)
    
    # df_features['Log_lat'] = np.log(df_features['Latitude (deg)'])  
    # df_features['Exp_lat'] = np.exp(df_features['Latitude (deg)'])  

    # df_features = extract_features(data_omni.drop(['Timestamp'],axis=1))
    # df_features = df_features.unstack().reset_index()
    # df_features.index = df_features['level_0'] + '_' + df_features['level_1']
    # df_features = df_features[[0]].T
    return df_features

def create_features_and_target(id,initial_states,omni = None,sat = None,path='phase_1/',predict_mean=True,training_mode=False,omni_path=None):

    if omni is None and omni_path is None:
        all_files_omni = os.listdir(path+"omni2/")
        file_omni = [i for i in all_files_omni if i.split('-')[1]==id][0]
        data_omni = pd.read_csv(path+'omni2/'+file_omni)
    elif omni_path is not None:
        data_omni = pd.read_csv(omni_path)
    else:
        data_omni = omni   
    
    if sat is None:
        if training_mode:
            all_files_sat = os.listdir(path+ "sat_density/")
            file_sat = [i for i in all_files_sat if i.split('-')[1]==id][0]
            data_sat = pd.read_csv(path+'sat_density/'+file_sat)
        else:
            data_sat = pd.DataFrame(create_pred_timestamps(initial_states['Timestamp'].loc[id]),columns=['Timestamp'])       
    else:
        data_sat = sat
    data_sat['Timestamp'] = pd.to_datetime(data_sat['Timestamp'],format='%Y-%m-%d %H:%M:%S')
    data_sat['File Id'] = id

    df_features_from_omni = pd.DataFrame(extract_features_from_omni(data_omni),index=pd.Index([id], name='File Id'))
    

    df_position = initial_states.loc[[id]].rename({'Timestamp':'Timestamp_initial'},axis=1)
    df_position['File Id'] = id
    df_features = pd.merge(df_features_from_omni,df_position,on='File Id')

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

def get_model_feat_import(model,n=15):
    data = pd.DataFrame({'feature_importance':model.get_feature_importance(), 
                'feature_names': model.feature_names_}).sort_values(by=['feature_importance'], 
                                                        ascending=False)
    return data.head(n)