import pandas as pd
from datetime import datetime, timedelta
import os
from tqdm import tqdm
import pickle
from atm_me import PersistenceMSIS, MSISPersistenceAtmosphere, PersistenceModel
tqdm.pandas()


def main():
    with open('omni.pickle', 'rb') as f:
        omni = pickle.load(f)
    with open('sat.pickle', 'rb') as f:
        sat = pickle.load(f)

    df_train = pd.read_csv(
        'C:/Users/isaac/Documents/Challenge_MIT_2/mit_challenge_2/phase_1/00000_to_02284-initial_states.csv')
    df_train = pd.concat([df_train, pd.read_csv(
        'C:/Users/isaac/Documents/Challenge_MIT_2/mit_challenge_2/phase_1/02285_to_02357-initial_states.csv')])
    df_train = pd.concat([df_train, pd.read_csv(
        'C:/Users/isaac/Documents/Challenge_MIT_2/mit_challenge_2/phase_1/02358_to_04264-initial_states.csv')])
    df_train = pd.concat([df_train, pd.read_csv(
        'C:/Users/isaac/Documents/Challenge_MIT_2/mit_challenge_2/phase_1/04265_to_05570-initial_states.csv')])
    df_train = pd.concat([df_train, pd.read_csv(
        'C:/Users/isaac/Documents/Challenge_MIT_2/mit_challenge_2/phase_1/05571_to_05614-initial_states.csv')])
    df_train = pd.concat([df_train, pd.read_csv(
        'C:/Users/isaac/Documents/Challenge_MIT_2/mit_challenge_2/phase_1/05615_to_06671-initial_states.csv')])
    # df_train = pd.concat([df_train,pd.read_csv('C:/Users/isaac/Documents/Challenge_MIT_2/mit_challenge_2/phase_1/06672_to_08118-initial_states.csv')])

    df_train['Timestamp'] = pd.to_datetime(df_train['Timestamp'])
    df_train = df_train.set_index('File ID')

    for id in tqdm(df_train.index[62:]):
        model = PersistenceModel(plot_trajectory=False)
        omni_data = omni[id].loc[:, [
            'Timestamp', 'f10.7_index', 'ap_index_nT']]
        omni_data['Timestamp'] = pd.to_datetime(omni_data['Timestamp'])
        omni_data = omni_data.ffill()
        states, densities = model(omni_data, df_train.loc[id].to_dict())
        predictions = model._convert_to_df(states, densities)
        predictions.to_csv(
            'phase_1/sat_density_pred/density_pred' + str(id) + '.csv')


if __name__ == "__main__":
    main()
