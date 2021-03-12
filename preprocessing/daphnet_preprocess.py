import os
import re

import numpy as np
import pandas as pd

DATA_DIR = 'data/raw/Daphnet/dataset_fog_release/dataset'

def get_daphnet_data():
    data_files = sorted(os.listdir(DATA_DIR))
    cols = ['Time', 'Ankle_acc_x', 'Ankle_acc_y', 'Ankle_acc_z', 'Thigh_acc_x', 'Thigh_acc_y', 'Thigh_acc_z', 'Trunk_acc_x', 'Trunk_acc_y', 'Trunk_acc_z', 'Label']
    
    complete_df = pd.DataFrame()
    
    for d_file in data_files:
        user, run = re.findall(r'\d+', d_file)
        
        data_arr = np.loadtxt(os.path.join(DATA_DIR, d_file), delimiter=' ')
        _df = pd.DataFrame(data=data_arr, columns=cols)
        _df['Subject'] = int(user)
        _df['Run'] = int(run)
        
        complete_df = pd.concat([complete_df, _df], ignore_index=True)
        
    complete_df = complete_df.sort_values(by=['Subject', 'Run'], ignore_index=True)
    complete_df.to_csv(os.path.join('data', 'processed', 'clean_daphnet_data.csv'), index=False)
    
    return complete_df