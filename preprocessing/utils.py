from preprocessing.opp_preprocess import *
from preprocessing.mhealth_preprocess import *
from preprocessing.pamap2_preprocess import *
from preprocessing.sliding_window import *



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
        if os.path.exists(os.path.join('data', 'processed', 'clean_opp.csv')):
            pass
        else:
            prepare_opp_data()
        df = pd.read_csv(os.path.join('data', 'processed', 'clean_opp.csv'))
        
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
    elif dataset == 'mhealth':
        metadata = yaml.load(metadata_file, Loader=yaml.FullLoader)['mhealth_preprocess']
        if not os.path.exists(os.path.join('data','raw', 'mhealth', 'MHEALTHDATASET')):
            print('Please download the mhealth dataset from uci using the provided script')
            return
        df = prepare_mhealth_data()
        train_df = df[df['subject_id'] != 9]
        test_df = df[df['subject_id'] == 9]

        SLIDING_WINDOW_LENGTH = metadata['sliding_win_len']
        SLIDING_WINDOW_STEP = metadata['sliding_win_stride']

        FEATURES = metadata['feature_list']

        X_train, y_train = create_windowed_dataset(train_df, FEATURES, 'label', window_size=SLIDING_WINDOW_LENGTH, stride = SLIDING_WINDOW_STEP)
        X_test, y_test = create_windowed_dataset(test_df, FEATURES, 'label', window_size=SLIDING_WINDOW_LENGTH, stride = SLIDING_WINDOW_STEP)

        N_WINDOW,N_TIMESTEP  = metadata['n_window'], metadata['n_timestep'] 
        X_train = X_train.reshape((X_train.shape[0], N_WINDOW, N_TIMESTEP, len(FEATURES)))
        X_test = X_test.reshape((X_test.shape[0], N_WINDOW, N_TIMESTEP, len(FEATURES)))

        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)
        
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
    else:
        print('Please enter a valid dataset name')
        return