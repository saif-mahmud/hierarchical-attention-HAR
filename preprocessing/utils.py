from preprocessing.opp_preprocess import *
from preprocessing.mhealth_preprocess import *
from preprocessing.pamap2_preprocess import *
from preprocessing.sliding_window import create_windowed_dataset
from preprocessing.mex_preprocess import get_mex_data
from preprocessing.realdisp_preprocess import get_realdisp_data
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import tensorflow as tf
import yaml
import os

def get_activity_dict(activity_map:dict, novel_classes:list):
    _activity_map = activity_map.copy()
    novel_map = dict()
    
    for activity_class in novel_classes:
        novel_map[activity_class] = activity_map[activity_class]
        _activity_map.pop(activity_class)
        
    return _activity_map, novel_map


def get_train_test_data(dataset, holdout=False):
    
    metadata_file = open('configs/metadata.yaml', mode='r')

    if dataset == 'opp':
        metadata = yaml.load(metadata_file, Loader=yaml.FullLoader)['opp_preprocess']
        FEATURES = [str(i) for i in range(77)]
        LOCO_LABEL_COL = 77
        MID_LABEL_COL = 78
        HI_LABEL_COL = 79
        SUBJECT_ID = 80
        RUN_ID = 81
        if not os.path.exists(os.path.join('data', 'processed', 'clean_opp.csv')):
            prepare_opp_data()
        df = pd.read_csv(os.path.join('data', 'processed', 'clean_opp.csv'))
        
        df = df[df[str(HI_LABEL_COL)] != 0]
        df[FEATURES] = df[FEATURES].interpolate(method='linear', axis=0)
        df = df.fillna(0)

        scaler = StandardScaler()
        df[FEATURES] = scaler.fit_transform(df[FEATURES])
        
        if holdout:
            NOVEL_CLASSES = [2, 4]
            holdout_data = df.loc[df[str(HI_LABEL_COL)].isin(NOVEL_CLASSES)]
            novel_data = holdout_data.copy().reset_index(drop=True)
            df = df.drop(holdout_data.copy().index)
            df = df.reset_index(drop=True)
        
        BENCHMARK_TEST= ((df[str(SUBJECT_ID)] == 2) | (df[str(SUBJECT_ID)] == 3)) & ((df[str(RUN_ID)] == 4) | (df[str(RUN_ID)] == 5))
        
        train_df = df[~ BENCHMARK_TEST]
        test_df = df[BENCHMARK_TEST]
        
        SLIDING_WINDOW_LENGTH = metadata['sliding_win_len']
        SLIDING_WINDOW_STEP = metadata['sliding_win_stride']
        N_WINDOW,N_TIMESTEP  = metadata['n_window'], metadata['n_timestep'] 
        
        X_train, y_train, m_labels_tr, loco_labels_tr = create_windowed_dataset_opp(train_df, FEATURES, str(HI_LABEL_COL), MID_LABEL_COL, LOCO_LABEL_COL, window_size=SLIDING_WINDOW_LENGTH, stride = SLIDING_WINDOW_STEP)
        X_test, y_test, m_labels_ts, loco_labels_ts = create_windowed_dataset_opp(test_df, FEATURES, str(HI_LABEL_COL),MID_LABEL_COL, LOCO_LABEL_COL, window_size=SLIDING_WINDOW_LENGTH, stride = SLIDING_WINDOW_STEP)
        if holdout:
            X_holdout, y_holdout, m_labels_holdout, loco_labels_holdout = create_windowed_dataset_opp(novel_data, FEATURES, str(HI_LABEL_COL),MID_LABEL_COL, LOCO_LABEL_COL, window_size=SLIDING_WINDOW_LENGTH, stride = SLIDING_WINDOW_STEP)
            X_holdout = X_holdout.reshape((X_holdout.shape[0], N_WINDOW, N_TIMESTEP, len(FEATURES)))
            y_holdout = tf.keras.utils.to_categorical(y_holdout-1)
        
        X_train = X_train.reshape((X_train.shape[0], N_WINDOW, N_TIMESTEP, len(FEATURES)))
        X_test = X_test.reshape((X_test.shape[0], N_WINDOW, N_TIMESTEP, len(FEATURES)))

        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)
        
        if holdout:
            return (X_train, y_train),  (X_test, y_test), (X_holdout, y_holdout)
        else:
            return (X_train, y_train),  (X_test, y_test)

    elif dataset == 'mhealth':
        metadata = yaml.load(metadata_file, Loader=yaml.FullLoader)['mhealth_preprocess']
        if not os.path.exists(os.path.join('data','raw', 'mhealth', 'MHEALTHDATASET')):
            print('Please download the mhealth dataset from uci using the provided script')
            return
        df = prepare_mhealth_data()
        if holdout:
            NOVEL_CLASSES = [1, 4, 8]
            holdout_data = df.loc[df['label'].isin(NOVEL_CLASSES)]
            novel_data = holdout_data.copy().reset_index(drop=True)

            df = df.drop(holdout_data.copy().index)
            df = df.reset_index(drop=True)

        train_df = df[df['subject_id'] != 9]
        test_df = df[df['subject_id'] == 9]

        SLIDING_WINDOW_LENGTH = metadata['sliding_win_len']
        SLIDING_WINDOW_STEP = metadata['sliding_win_stride']
        N_WINDOW,N_TIMESTEP  = metadata['n_window'], metadata['n_timestep'] 

        FEATURES = metadata['feature_list']

        X_train, y_train = create_windowed_dataset(train_df, FEATURES, 'label', window_size=SLIDING_WINDOW_LENGTH, stride = SLIDING_WINDOW_STEP)
        X_test, y_test = create_windowed_dataset(test_df, FEATURES, 'label', window_size=SLIDING_WINDOW_LENGTH, stride = SLIDING_WINDOW_STEP)
        if holdout:
            X_holdout, y_holdout = create_windowed_dataset(novel_data,  FEATURES, 'label', window_size=SLIDING_WINDOW_LENGTH, stride = SLIDING_WINDOW_STEP)
            X_holdout = X_holdout.reshape((X_holdout.shape[0], N_WINDOW, N_TIMESTEP, len(FEATURES)))
            y_holdout = tf.keras.utils.to_categorical(y_holdout-1)

        X_train = X_train.reshape((X_train.shape[0], N_WINDOW, N_TIMESTEP, len(FEATURES)))
        X_test = X_test.reshape((X_test.shape[0], N_WINDOW, N_TIMESTEP, len(FEATURES)))

        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)
        
        if holdout:
            return (X_train, y_train),  (X_test, y_test), (X_holdout, y_holdout)
        else:
            return (X_train, y_train),  (X_test, y_test)

    elif dataset == 'pamap2':
        metadata = yaml.load(metadata_file, Loader=yaml.FullLoader)['pamap2_preprocess']
        file_path = os.path.join('data', 'processed', 'pamap2_106.h5')
        if not os.path.exists(file_path):
            train_test_files = metadata['file_list']
            use_columns = metadata['columns_list']
            output_file_name = file_path
            label_to_id = metadata['label_to_id']
            read_dataset_pamap2(train_test_files, use_columns, output_file_name, label_to_id)
        
        (train_x, train_y), (val_x, val_y), (test_x, test_y) = preprocess_pamap2(file_path, downsample=False)

        SLIDING_WINDOW_LENGTH = metadata['sliding_win_len']
        SLIDING_WINDOW_STEP = metadata['sliding_win_stride']

        X_train, y_train = create_windowed_dataset(None, None, None, X=train_x, y=train_y, window_size=SLIDING_WINDOW_LENGTH, stride = SLIDING_WINDOW_STEP)
        X_test, y_test = create_windowed_dataset(None, None, None, X=test_x, y=test_y, window_size=SLIDING_WINDOW_LENGTH, stride = SLIDING_WINDOW_STEP)

        N_WINDOW,N_TIMESTEP  = metadata['n_window'], metadata['n_timestep']
        X_train = X_train.reshape((X_train.shape[0], N_WINDOW, N_TIMESTEP, 18))
        X_test = X_test.reshape((X_test.shape[0], N_WINDOW, N_TIMESTEP, 18))
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=19)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=19)

        return (X_train, y_train),  (X_test, y_test)
    
    elif dataset == 'mex':
        metadata = yaml.load(metadata_file, Loader=yaml.FullLoader)['mex_preprocess']

        if os.path.exists(metadata['data_dir']):
            df = pd.read_csv(metadata['data_dir'])
        else:
            df = get_mex_data()

        FEATURES = metadata['FEATURES']
        FEATURES_THIGH = metadata['FEATURES_THIGH']
        FEATURES_WRIST = metadata['FEATURES_WRIST']
        LABELS = metadata['LABELS']

        NOVEL_CLASSES = metadata['NOVEL_CLASSES']

        WINDOW_SIZE = metadata['sliding_win_len']
        STRIDE = metadata['sliding_win_stride']

        N_WINDOW = metadata['n_window']
        N_TIMESTEP = metadata['n_timestep']

        scaler = StandardScaler()
        df[FEATURES] = scaler.fit_transform(df[FEATURES])

        if holdout:
            holdout_data = df.loc[df[LABELS].isin(NOVEL_CLASSES)]
            novel_data = holdout_data.copy().reset_index(drop=True)

            df = df.drop(holdout_data.copy().index)
            df = df.reset_index(drop=True)

            X_holdout, y_holdout = create_windowed_dataset(novel_data, FEATURES, class_label=LABELS, window_size=WINDOW_SIZE, stride = STRIDE)
            X_holdout = X_holdout.reshape((X_holdout.shape[0], N_WINDOW, N_TIMESTEP, 6))
            y_holdout = tf.keras.utils.to_categorical(y_holdout-1)

        subjects = set(range(1, 31))
        test_sub = set(range(11, 15))
        train_sub = subjects - test_sub

        train_df = df[df['subject_id'].isin(train_sub)]
        test_df = df[df['subject_id'].isin(test_sub)]

        X_train, y_train = create_windowed_dataset(train_df,FEATURES, class_label=LABELS, window_size=WINDOW_SIZE, stride = STRIDE)
        X_test, y_test = create_windowed_dataset(test_df,FEATURES, class_label=LABELS, window_size=WINDOW_SIZE, stride = STRIDE)

        X_train = X_train.reshape((X_train.shape[0], N_WINDOW, N_TIMESTEP, 6))
        X_test = X_test.reshape((X_test.shape[0], N_WINDOW, N_TIMESTEP, 6))
        

        y_train = tf.keras.utils.to_categorical(y_train-1)
        y_test = tf.keras.utils.to_categorical(y_test-1)

        if holdout:
            return (X_train, y_train),  (X_test, y_test), (X_holdout, y_holdout)
        else:
            return (X_train, y_train),  (X_test, y_test)

    elif dataset == 'realdisp':
        metadata = yaml.load(metadata_file, Loader=yaml.FullLoader)['realdisp_preprocess']

        if os.path.exists(metadata['data_dir']):
            df = pd.read_csv(metadata['data_dir'])
        else:
            df = get_realdisp_data()

        df = df[df['DISPLACEMENT'].isin(['ideal', 'self'])]
        df = df.sort_values(by=['SUBJECT', 'LABEL', 'TIME_SECOND', 'TIME_MICROSECOND'], ignore_index=True)

        SENSOR_PLACEMENT = metadata['SENSOR_PLACEMENT']
        SENSOR_LIST = metadata['SENSOR_LIST']
        FEATURES = list()
        for loc in SENSOR_PLACEMENT:
            for sensor in SENSOR_LIST:
                FEATURES.append(str(loc + '_' + sensor))
        LABELS = metadata['LABELS']

        scaler = StandardScaler()
        df[FEATURES] = scaler.fit_transform(df[FEATURES])

        if holdout:
            holdout_data = df.loc[df['LABEL'].isin(NOVEL_CLASSES)]
            novel_data = holdout_data.copy().reset_index(drop=True)

            df = df.drop(holdout_data.copy().index)
            df = df.reset_index(drop=True)
            X_holdout, y_holdout = create_windowed_dataset(novel_data, FEATURES, class_label=LABELS, window_size=WINDOW_SIZE, stride = STRIDE)
            X_holdout = X_holdout.reshape((X_holdout.shape[0], N_WINDOW, N_TIMESTEP, len(FEATURES)))
            y_holdout = tf.keras.utils.to_categorical(y_holdout-1, num_classes=33)

        train_sub = set(range(1, 18))
        test_sub = set([7])
        train_sub = train_sub - test_sub

        train_df = df[df['SUBJECT'].isin(train_sub)]
        test_df = df[df['SUBJECT'].isin(test_sub)]

        NOVEL_CLASSES = metadata['NOVEL_CLASSES']

        WINDOW_SIZE = metadata['sliding_win_len']
        STRIDE = metadata['sliding_win_stride']

        N_WINDOW = metadata['n_window']
        N_TIMESTEP = metadata['n_timestep']

        X_train, y_train = create_windowed_dataset(train_df,FEATURES, class_label=LABELS, window_size=WINDOW_SIZE, stride = STRIDE)
        X_test, y_test = create_windowed_dataset(test_df,FEATURES, class_label=LABELS, window_size=WINDOW_SIZE, stride = STRIDE)

        X_train = X_train.reshape((X_train.shape[0], N_WINDOW, N_TIMESTEP, len(FEATURES)))
        X_test = X_test.reshape((X_test.shape[0], N_WINDOW, N_TIMESTEP, len(FEATURES)))

        y_train = tf.keras.utils.to_categorical(y_train - 1, num_classes=33)
        y_test = tf.keras.utils.to_categorical(y_test - 1, num_classes=33)

        if holdout:
            return (X_train, y_train),  (X_test, y_test), (X_holdout, y_holdout)
        else:
            return (X_train, y_train),  (X_test, y_test)
        

            
