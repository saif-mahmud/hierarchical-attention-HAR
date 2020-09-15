import csv
import os
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


def create_windowed_dataset(df, features, class_label = 'label', window_size=24, stride = 12):
    X = df[features].values
    y = df[class_label].values
    segments = []
    labels = []
    seg_start= 0
    seg_end = window_size
    while seg_end <= len(X):
        if len(np.unique(y[seg_start:seg_end])) == 1:
            segments.append(X[seg_start:seg_end])
            labels.append(y[seg_start])

            seg_start += stride
            seg_end = seg_start + window_size

        else:
            current_label = y[seg_start]
            for i in range(seg_start, seg_end):
                if y[i] != current_label:
                    seg_start = i
                    seg_end = seg_start + window_size
                    break

    return np.asarray(segments).astype(np.float32), np.asarray(labels)

# def reshape_for_session(X, n_window, n_timesteps):
#     return 
    


def create_windowed_dataset_opp(df, features, class_label, MID_LABEL_COL, LOCO_LABEL_COL, window_size=24, stride = 12):
    X = df[features].values
    y = df[class_label].values
    segments = []
    labels = []
    seg_start= 0
    seg_end = window_size
    mid_labels = []
    loco_labels = []
    while seg_end <= len(X):
        if len(np.unique(y[seg_start:seg_end])) == 1:
            segments.append(X[seg_start:seg_end])
            labels.append(y[seg_start]) 
            mid_labels.append(df[str(MID_LABEL_COL)].values[seg_start:seg_end])
            loco_labels.append(df[str(LOCO_LABEL_COL)].values[seg_start:seg_end])

            seg_start += stride
            seg_end = seg_start + window_size

        else:
            current_label = y[seg_start]
            for i in range(seg_start, seg_end):
                if y[i] != current_label:
                    seg_start = i
                    seg_end = seg_start + window_size
                    break

    return np.asarray(segments).astype(np.float32), np.asarray(labels), mid_labels, loco_labels


def readOpportunityFiles(filelist, cols, mid_label_to_id, hi_label_to_id, loco_label_to_id):
    data = []
    mid_labels = []
    hi_labels = []
    loco_labels = []
    subject_mapping = []
    base_path = os.path.join('data','raw', 'opp', 'OpportunityUCIDataset', 'dataset')
    assert os.path.exists(base_path), "Please download the dataset first using the script"

    for i, filename in enumerate(filelist):
        with open(os.path.join(base_path, filename), 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            subject_info = [int(filename[1]), filename[6:7]]
            for line in reader:
                elem = []
                for ind in cols:
                    elem.append(line[ind])
                if sum([x != 'NaN' for x in elem]) > 40:
                    data.append([float(x) for x in elem[:-3]])
                    mid_labels.append(mid_label_to_id[elem[-1]])
                    hi_labels.append(hi_label_to_id[elem[-2]])
                    loco_labels.append(loco_label_to_id[elem[-3]])
                    subject_mapping.append(subject_info)

    return np.asarray(data), np.asarray(mid_labels, dtype=int), np.asarray(hi_labels, dtype=int), np.asarray(loco_labels, dtype=int), np.asarray(subject_mapping)


def prepare_opp_data():
    metadata_file = open('configs/metadata.yaml', mode='r')
    metadata = yaml.load(metadata_file, Loader=yaml.FullLoader)['opp_preprocess']
    file_list_nodrill = metadata['file_list']
    
    mid_label_to_id = metadata['mid_label_to_id']
    hi_label_to_id = metadata['hi_label_to_id']
    loco_label_to_id = metadata['loco_label_to_id']

    cols = metadata['columns_list']

    selected_cols = np.asarray(cols)-1
    
    data, mid_labels, hi_labels, loco_labels, subject_mapping = readOpportunityFiles(file_list_nodrill, selected_cols, mid_label_to_id, hi_label_to_id, loco_label_to_id)
    
    shp = data.shape[0]
    combined = pd.DataFrame(np.hstack((data.astype(np.float32), loco_labels.reshape((shp, 1)), mid_labels.reshape((shp, 1)), hi_labels.reshape((shp, 1)), subject_mapping)))
    combined.to_csv(os.path.join('data', 'clean', 'clean_opp.csv'), index=False)
    
def get_train_test_data(dataset):
    metadata_file = open('configs/metadata.yaml', mode='r')
    if dataset == 'opp':
        metadata = yaml.load(metadata_file, Loader=yaml.FullLoader)['opp_preprocess']
        FEATURES = [str(i) for i in range(77)]
        LOCO_LABEL_COL = 77
        MID_LABEL_COL = 78
        HI_LABEL_COL = 79
        SUBJECT_ID = 80
        RUN_ID = 81
        if os.path.exists(os.path.join('data', 'clean', 'clean_opp.csv')):
            pass
        else:
            prepare_opp_data()
        df = pd.read_csv(os.path.join('data', 'clean', 'clean_opp.csv'))
        
        df[FEATURES] = df[FEATURES].interpolate(method='linear', axis=0)
        df = df.fillna(0)

        scaler = StandardScaler()
        df[FEATURES] = scaler.fit_transform(df[FEATURES])
        
        BENCHMARK_TEST= ((df[str(SUBJECT_ID)] == 2) | (df[str(SUBJECT_ID)] == 3)) & ((df[str(RUN_ID)] == 4) | (df[str(RUN_ID)] == 5))
        
        train_df = df[~ BENCHMARK_TEST]
        test_df = df[BENCHMARK_TEST]
        
        SLIDING_WINDOW_LENGTH = metadata['sliding_win_len']
        SLIDING_WINDOW_STEP = metadata['sliding_win_stride']
        
        X_train, y_train, m_labels_tr, loco_labels_tr = create_windowed_dataset_opp(train_df, FEATURES, str(HI_LABEL_COL), MID_LABEL_COL, LOCO_LABEL_COL, window_size=SLIDING_WINDOW_LENGTH, stride = SLIDING_WINDOW_STEP)
        X_test, y_test, m_labels_ts, loco_labels_ts = create_windowed_dataset_opp(test_df, FEATURES, str(HI_LABEL_COL),MID_LABEL_COL, LOCO_LABEL_COL, window_size=SLIDING_WINDOW_LENGTH, stride = SLIDING_WINDOW_STEP)
        
        
        N_WINDOW,N_TIMESTEP  = metadata['n_window'], metadata['n_timestep'] 
        X_train = X_train.reshape((X_train.shape[0], N_WINDOW, N_TIMESTEP, len(FEATURES)))
        X_test = X_test.reshape((X_test.shape[0], N_WINDOW, N_TIMESTEP, len(FEATURES)))

        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)
        
        return (X_train, y_train),  (X_test, y_test)
    else:
        pass
    
