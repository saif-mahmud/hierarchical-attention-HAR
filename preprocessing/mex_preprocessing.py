import glob
import os
import time
import datetime
import pandas as pd
import numpy as np

DATA_DIR = '../data/raw/mex'
THIGH_ACCEL = 'act'
WRIST_ACCEL = 'acw'

def quantize_time(timestamp, quantization_level=2):
    m_sec = str(timestamp.microsecond)
    q_msec = m_sec[:quantization_level]

    timestamp = timestamp.replace(microsecond=(int(q_msec) * (10 ** (6 - quantization_level))))

    return timestamp

def get_activity(file_name:str):
    activivity = int(file_name.strip().split('_')[0])

    return activivity

def get_clean_data(df:pd.DataFrame, method='bfill', drop=False):
    if not drop:
        clean_df = df.copy().fillna(method=method)
    else:
        clean_df = df.copy().dropna()
    
    clean_df = clean_df.sort_values(by='timestamp', ignore_index=True)
    clean_df = clean_df.dropna() # Remove null value in last row

    return clean_df

def get_mex_data():

    complete_df = pd.DataFrame()

    SUBJECT_LIST = ['%02d' % x for x in range(1, 31)]

    for subj in SUBJECT_LIST:
        accel_t_dir = os.path.join(DATA_DIR, THIGH_ACCEL, subj)
        accel_w_dir = os.path.join(DATA_DIR, WRIST_ACCEL, subj)

        accel_t_files = sorted(os.listdir(accel_t_dir))
        accel_w_files = sorted(os.listdir(accel_w_dir))

        for i in range(len(accel_t_files)):
            accel_t = os.path.join(accel_t_dir, accel_t_files[i])
            accel_w = os.path.join(accel_w_dir, accel_w_files[i])

            df_t = pd.read_csv(accel_t, header=None, names=['timestamp', 'act_x', 'act_y', 'act_z'])
            df_w = pd.read_csv(accel_w, header=None, names=['timestamp', 'acw_x', 'acw_y', 'acw_z'])

            df_t['timestamp'] = pd.to_datetime(df_t['timestamp'])
            df_w['timestamp'] = pd.to_datetime(df_w['timestamp'])

            df_t['timestamp'] = df_t['timestamp'].apply(quantize_time)
            df_w['timestamp'] = df_w['timestamp'].apply(quantize_time)

            df_t = df_t.groupby(['timestamp'], as_index=False)['act_x', 'act_y', 'act_z'].mean()
            df_w = df_w.groupby(['timestamp'], as_index=False)['acw_x', 'acw_y', 'acw_z'].mean()

            merged = pd.merge(df_t, df_w, how='outer', on=['timestamp', 'timestamp'], sort=True)
            merged['subject_id'] = int(subj)
            merged['activity'] = get_activity(accel_t_files[i])
            
            complete_df = pd.concat([complete_df, merged], ignore_index=True)

    complete_df = complete_df.sort_values(by='timestamp', ignore_index=True)
    complete_df = get_clean_data(complete_df)
    complete_df.to_csv('../data/processed/clean_mex_data.csv', index=False)

    return complete_df