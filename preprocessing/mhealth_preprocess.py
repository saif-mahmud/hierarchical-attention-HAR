import csv
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.preprocessing import StandardScaler


def prepare_mhealth_data():
    metadata_file = open('configs/metadata.yaml', mode='r')
    metadata = yaml.load(metadata_file, Loader=yaml.FullLoader)['mhealth_preprocess']

    columns = metadata['columns_list']
    base_path = os.path.join('data','raw', 'mhealth', 'MHEALTHDATASET')

    df = pd.DataFrame()
    for i in range(1,11):
        temp = pd.read_csv(os.path.join(base_path, f'mHealth_subject{i}.log'), sep='\t', header=None)
        temp.columns= columns
        temp['subject_id'] = i
        df = df.append(temp, ignore_index=True)
    df = df.reset_index(drop=True)
    df = df[df['label'] != 0]
    columns.append('subject_id')
    FEATURES = metadata['feature_list']

    df = df.fillna(0)
    scaler = StandardScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])

    return df
