import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import yaml

sys.path.append("../")

from preprocessing.opp_preprocess import *
from model.hierarchical_self_attention_model import HSA_model_session_guided_window

def session_length_variation():
    activity_list = ['Other', 'Open Door 1', 'Open Door 2', 'Close Door 1',
                    'Close Door 2', 'Open Fridge', 'Close Fridge', 
                    'Open Dishwasher', 'Close Dishwasher', 'Open Drawer 1', 
                    'Close Drawer 1', 'Open Drawer 2', 'Close Drawer 2', 
                    'Open Drawer 3', 'Close Drawer 3', 'Clean Table', 
                    'Drink from Cup', 'Toggle Switch']
    
    DATA_PATH = '/home/hariub/data/HAR/processed/clean_opp_nodrill.csv'
    df = pd.read_csv(DATA_PATH)
    
    FEATURES = [str(i) for i in range(77)]
    LOCO_LABEL_COL = 77
    MID_LABEL_COL = 78
    HI_LABEL_COL = 79
    SUBJECT_ID = 80
    RUN_ID = 81
    
    df[FEATURES] = df[FEATURES].interpolate(method='linear', axis=0)
    df = df.fillna(0)
    scaler = StandardScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])

    BENCHMARK_TEST = ((df[str(SUBJECT_ID)] == 2) | (df[str(SUBJECT_ID)] == 3)) & (
                (df[str(RUN_ID)] == 4) | (df[str(RUN_ID)] == 5))

    train_df = df[~ BENCHMARK_TEST]
    test_df = df[BENCHMARK_TEST]
    
    session_len_list = [20, 30, 40, 50, 60, 90, 120 ]
    for session_len in session_len_list:
        SLIDING_WINDOW_LENGTH = session_len
        SLIDING_WINDOW_STEP = SLIDING_WINDOW_LENGTH // 2
        N_WINDOW, N_TIMESTEP = 10, SLIDING_WINDOW_LENGTH //10

        X_train, y_train, m_labels_tr, loco_labels_tr = create_windowed_dataset_opp(train_df, FEATURES, str(
            MID_LABEL_COL), MID_LABEL_COL, LOCO_LABEL_COL, window_size=SLIDING_WINDOW_LENGTH, stride=SLIDING_WINDOW_STEP)
        X_test, y_test, m_labels_ts, loco_labels_ts = create_windowed_dataset_opp(test_df, FEATURES, str(
            MID_LABEL_COL), MID_LABEL_COL, LOCO_LABEL_COL, window_size=SLIDING_WINDOW_LENGTH, stride=SLIDING_WINDOW_STEP)

        X_train = X_train.reshape(
            (X_train.shape[0], N_WINDOW, N_TIMESTEP, len(FEATURES)))
        X_test = X_test.reshape(
            (X_test.shape[0], N_WINDOW, N_TIMESTEP, len(FEATURES)))

        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)
        
        y_train_mid = np.repeat(np.expand_dims(y_train, axis=1), repeats=N_WINDOW, axis=1)
        # y_val_mid = np.repeat(np.expand_dims(y_val, axis=1), repeats=N_WINDOW, axis=1)
        y_test_mid = np.repeat(np.expand_dims(y_test, axis=1), repeats=N_WINDOW, axis=1)
        
        hparam_file = open('../configs/hyperparameters.yaml', mode='r')
        hyperparameters = yaml.load(hparam_file, Loader=yaml.FullLoader)
        DATASET_NAME = 'opp'
        hparams_all = hyperparameters['HSA_model']
        hparams = hparams_all[DATASET_NAME]

        hparams['n_window'], hparams['n_timesteps'], hparams['n_features'], hparams['n_outputs'] = X_train.shape[1], X_train.shape[2], X_train.shape[3], y_train.shape[1]
        hparams['n_outputs_window']=  y_train.shape[1]
        
        tf.keras.backend.clear_session()
#         gpus = tf.config.experimental.list_physical_devices('GPU')
#         if gpus:
#           try:
#             # Currently, memory growth needs to be the same across GPUs
#             for gpu in gpus:
#               tf.config.experimental.set_memory_growth(gpu, True)
#             logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#             print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#           except RuntimeError as e:
#             # Memory growth must be set before GPUs have been initialized
#             print(e)

        device_list = ['/gpu:'+str(i) for i in range(5, 8)]
        strategy = tf.distribute.MirroredStrategy(devices=device_list)
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            model = HSA_model_session_guided_window(**hparams).get_compiled_model()
            
        checkpoint_filepath = f"opp_checkpoints/session_var/cp-{session_len}" + "-{epoch:04d}.ckpt"
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                       save_weights_only=True)
        
        model.fit(X_train, [y_train_mid, y_train], batch_size=len(device_list) * 128, epochs=40, validation_split=0.1, use_multiprocessing=True, callbacks=[model_checkpoint_callback], verbose=0)
        
        model_from_ckpt = HSA_model_session_guided_window(**hparams).get_compiled_model()
        model_from_ckpt.load_weights(f"opp_checkpoints/session_var/cp-{session_len}" + "-0040.ckpt")
        pred_mid, pred_sess =model_from_ckpt.predict(X_test, batch_size= 64)
        
        if not os.path.exists('result'):
            os.mkdir('result')
            
        out_res_s = open(os.path.join('result',
                                    str('session_' + str(session_len).zfill(3) + '_session' + '.txt')), 'w')
        
        
#         print(classification_report(np.argmax(y_test, axis=1), np.argmax(pred_sess, axis=1), target_names=activity_list))
        print(classification_report(np.argmax(y_test, axis=1), np.argmax(pred_sess, axis=1), target_names=activity_list), file=out_res_s)
        
        confm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(pred_sess, axis=1), normalize='pred')
        df_cm = pd.DataFrame(confm, index=activity_list, columns=activity_list)
        plt.figure(figsize = (20,15))
        sns.heatmap(df_cm, annot=True, fmt='0.3f', cmap="YlGnBu")
        
        if not os.path.exists('figures'):
            os.mkdir('figures')
        out_fig = 'session_' + str(session_len).zfill(3) + '.png'
        plt.savefig(os.path.join('figures', out_fig))
        
        out_res_w = open(os.path.join('result',
                                    str('session_' + str(session_len).zfill(3) + '_window' + '.txt')), 'w')
#         print(classification_report(np.argmax(y_test_mid.reshape(-1, 18), axis=1), np.argmax(pred_mid.reshape(-1, 18), axis=1), target_names=activity_list))
        print(classification_report(np.argmax(y_test_mid.reshape(-1, 18), axis=1), np.argmax(pred_mid.reshape(-1, 18), axis=1), target_names=activity_list), file=out_res_w)
    
    

if __name__ == "__main__":
    import logging
    tf.get_logger().setLevel(logging.ERROR)
    session_length_variation()