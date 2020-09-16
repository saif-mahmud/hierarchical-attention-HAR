import tensorflow as tf
import numpy as np
import pandas as pd
import yaml
import os
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, auc, roc_curve, f1_score, accuracy_score
from matplotlib import pyplot as plt
from preprocessing.sliding_window import create_windowed_dataset
from model.hierarchical_self_attention_model import HSA_model

hparam_file = open('configs/hyperparameters.yaml', mode='r')
hyperparameters = yaml.load(hparam_file, Loader=yaml.FullLoader)

metadata_file = open('configs/metadata.yaml', mode='r')


def run_loso_experiment(dataset: str, df: pd.DataFrame, activity_map: dict, sub_start=1):

    metadata = yaml.load(metadata_file, Loader=yaml.FullLoader)[
        dataset + '_preprocess']

    FEATURES = metadata['FEATURES']
    LABELS = metadata['LABELS']

    WINDOW_SIZE = metadata['sliding_win_len']
    STRIDE = metadata['sliding_win_stride']

    N_WINDOW = metadata['n_window']
    N_TIMESTEP = metadata['n_timestep']

    for i in range(sub_start, df['SUBJECT'].nunique() + 1):

        train_sub = set(range(1, df['SUBJECT'].nunique() + 1))
        test_sub = set([i])
        train_sub = train_sub - test_sub

        print('TEST SUBJECTS : ', test_sub)
        print('TRAIN SUBJECTS : ', train_sub)

        train_df = df[df['SUBJECT'].isin(train_sub)]
        test_df = df[df['SUBJECT'].isin(test_sub)]

        X_train, y_train = create_windowed_dataset(
            train_df, FEATURES, class_label=LABELS, window_size=WINDOW_SIZE, stride=STRIDE)
        X_test, y_test = create_windowed_dataset(
            test_df, FEATURES, class_label=LABELS, window_size=WINDOW_SIZE, stride=STRIDE)

        X_train = X_train.reshape(
            (X_train.shape[0], N_WINDOW, N_TIMESTEP, len(FEATURES)))
        X_test = X_test.reshape(
            (X_test.shape[0], N_WINDOW, N_TIMESTEP, len(FEATURES)))

        y_train = tf.keras.utils.to_categorical(
            y_train - 1, num_classes=df['LABEL'].nunique())
        y_test = tf.keras.utils.to_categorical(
            y_test - 1, num_classes=df['LABEL'].nunique())

        tf.keras.backend.clear_session()

        n_window, n_timesteps, n_features, n_outputs = X_train.shape[
            1], X_train.shape[2], X_train.shape[3], y_train.shape[1]

        hsa_model = HSA_model(hyperparameters['HSA_model']['modality_indices'][dataset], n_window, n_timesteps, n_features, n_outputs, d_model=hyperparameters['HSA_model']['d_model'],
                              num_heads=hyperparameters['HSA_model']['num_heads'], dff=hyperparameters['HSA_model']['dff'], dropout_rate=hyperparameters['HSA_model']['dropout']).get_model()

        hsa_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(
            lr=hyperparameters['train']['learning_rate']), metrics='accuracy')

        hsa_model.fit(X_train, y_train, epochs=hyperparameters['train']['epochs'], batch_size=hyperparameters['train']
                      ['batch_size'], verbose=1, validation_split=hyperparameters['train']['val_split'])

        pred = hsa_model.predict(
            X_test, batch_size=hyperparameters['train']['batch_size'])

        out_res = open(os.path.join('result/realdisp',
                                    str('subject___' + str(i).zfill(2) + '.txt')), 'w')
        print(classification_report(np.argmax(y_test, axis=1), np.argmax(pred, axis=1), labels=np.unique(np.argmax(y_test, axis=1)),
                                    target_names=list(activity_map.values()), zero_division=1), file=out_res)
        print(classification_report(np.argmax(y_test, axis=1), np.argmax(pred, axis=1),
                                    labels=np.unique(np.argmax(y_test, axis=1)), target_names=list(activity_map.values()), zero_division=1))
        out_res.close()

        confm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(
            pred, axis=1), normalize='true')
        activity_list = np.array(list(activity_map.values()))[np.union1d(
            np.argmax(y_test, axis=1), np.argmax(pred, axis=1))]
        df_cm = pd.DataFrame(confm, index=activity_list, columns=activity_list)
        plt.figure(figsize=(16, 16))
        sns.heatmap(df_cm, annot=True, linewidths=0.05,
                    linecolor='blue', cmap="PuBu")

        out_fig = 'subject___' + str(i).zfill(2) + '.png'
        plt.savefig(os.path.join('figures/realdisp', out_fig))

        plt.show()
