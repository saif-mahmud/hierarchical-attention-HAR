import os
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = '../data/raw/realdisp'
DATA_FILES = sorted(os.listdir(DATA_DIR))

SENSOR_PLACEMENT = ['RLA', 'RUA', 'BACK', 'LUA', 'LLA', 'RC', 'RT', 'LT', 'LC']
SENSOR_LIST = ['ACC_X', 'ACC_Y', 'ACC_Z', 'GYR_X', 'GYR_Y', 'GYR_Z', 
               'MAG_X', 'MAG_Y', 'MAG_Z', 'QUAT_1', 'QUAT_2', 'QUAT_3', 'QUAT_4']

DATA_COLUMNS = ['TIME_SECOND', 'TIME_MICROSECOND']

SENSOR_READINGS = list()

for loc in SENSOR_PLACEMENT:
    for sensor in SENSOR_LIST:
        SENSOR_READINGS.append(str(loc + '_' + sensor))
        
DATA_COLUMNS.extend(SENSOR_READINGS)
DATA_COLUMNS.append('LABEL')

def get_metadata(filename:str):
    _name = filename.split('.')[0]
    
    subject = int(''.join(filter(str.isdigit, _name.split('_')[0])))
    disp = ''.join(i for i in _name.split('_')[1] if not i.isdigit())
    
    return subject, disp

def get_realdisp_data():
    merged_df = pd.DataFrame()
    
    for d_file in tqdm(DATA_FILES):
        subject, disp = get_metadata(d_file)
        
        data = np.loadtxt(os.path.join(DATA_DIR, d_file))
        df = pd.DataFrame.from_records(data)
        
        df = df[df[119] != 0.0].reset_index(drop=True)
        df.columns = DATA_COLUMNS
        
        df['SUBJECT'] = subject
        df['DISPLACEMENT'] = disp
        
        merged_df = pd.concat([merged_df, df], ignore_index=True)
        
    merged_df = merged_df.sort_values(by=['TIME_SECOND', 'TIME_MICROSECOND'], ignore_index=True)
    
    idx = merged_df[(merged_df['TIME_SECOND'] == 0.0) & (merged_df['TIME_MICROSECOND'] == 0.0)].index
    merged_df = merged_df.drop(idx, inplace=False).reset_index(drop=True)
    
    merged_df.to_csv('../data/processed/clean_realdisp_data.csv', index=False)
    return merged_df
