import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tabulate import tabulate

from experiments.novelty_detection import novelty_detection_exp
from preprocessing.utils import get_train_test_data, get_activity_dict
from train import train_model

warnings.filterwarnings("ignore")
# tf.get_logger().setLevel('INFO')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


hparam_file = open('configs/hyperparameters.yaml', mode='r')
hyperparameters = yaml.load(hparam_file, Loader=yaml.FullLoader)

metadata_file = open('configs/metadata.yaml', mode='r')

dataset = str(sys.argv[1])
try:
    arg2 = str(sys.argv[2])
except IndexError:
    arg2 = None
try:
    arg3 = str(sys.argv[3])
except IndexError:
    arg3 = None
try:
    arg4 = str(sys.argv[4])
except IndexError:
    arg4 = None


def test_hsa_model(dataset, arg2=None, arg3=None, arg4=None):
    print(tabulate([['Hierarchical Self Attention Based Human Activity Recognition and Novel Activity Detection']], [
    ], tablefmt="fancy_grid"))
    print('[PREPROCESSING AND LOADING DATA] ...')
    if (arg2 == 'use_pretrained') or (arg3 == 'use_pretrained') or (arg4 == 'use_pretrained'):
        if (arg2 == 'include_novelty_exp') or (arg3 == 'include_novelty_exp') or (arg4 == 'include_novelty_exp'):
            print('Novelty experiment using pretrained weights not currently supported')
            return
        else:
            (X_train, y_train), (X_test, y_test) = get_train_test_data(
                dataset=dataset, holdout=False)
            if os.path.exists(os.path.join('saved_models', dataset)):
                model_hsa = train_model(
                    dataset, (X_train, y_train), train_hsa=False)
                model_hsa.load_weights(os.path.join('saved_models', dataset, dataset))
            else:
                print('Pretrained weights not available, starting training')
                model_hsa = train_model(
                    dataset, (X_train, y_train), train_hsa=True)
            # if
            # model_hsa.load
    else:
        if (arg2 == 'include_novelty_exp') or (arg3 == 'include_novelty_exp') or (arg4 == 'include_novelty_exp'):
            (X_train, y_train),  (X_test, y_test), (X_holdout,
                                                    y_holdout) = get_train_test_data(dataset=dataset, holdout=True)
            model_hsa, model_vae = train_model(
                dataset, (X_train, y_train), train_vae=True)
        else:
            (X_train, y_train), (X_test, y_test) = get_train_test_data(
                dataset=dataset, holdout=False)
            model_hsa = train_model(dataset, (X_train, y_train))

    if (arg2 == 'save_weights') or (arg3 == 'save_weights') or (arg4 == 'save_weights'):
        if not os.path.exists(os.path.join('saved_models', dataset)):
            os.mkdir(os.path.join('saved_models', dataset))
        model_hsa.save_weights(os.path.join('saved_models', dataset, dataset))

    pred = model_hsa.predict(
        X_test, batch_size=hyperparameters['test']['batch_size'])

    activity_map = json.load(
        open(os.path.join('data', 'activity_maps', dataset + '_activity.json')))

    if(arg2 == 'include_novelty_exp') or (arg3 == 'include_novelty_exp') or (arg4 == 'include_novelty_exp'):
        metadata = yaml.load(metadata_file, Loader=yaml.FullLoader)[
            dataset + '_preprocess']
        NOVEL_CLASSES = metadata['NOVEL_CLASSES']
        activity_map, novel_map = get_activity_dict(
            activity_map, NOVEL_CLASSES)
        print('\nNOVEL / UNSEEN ACTIVITIES: ', novel_map)
        print()

    activity_names = list(activity_map.values())

    print(classification_report(np.argmax(y_test, axis=1), np.argmax(pred, axis=1),
                                labels=range(len(activity_names)), target_names=activity_names, zero_division=1))
    # out_res = open(os.path.join('result', dataset + '_classification_report.txt'))
    # print(classification_report(np.argmax(y_test, axis=1), np.argmax(pred, axis=1),labels = range(len(activity_names)), target_names=activity_names, zero_division=1), file=out_res)
    # out_res.close()

    confm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(
        pred, axis=1), labels=range(len(activity_names)))
    print(confm)

    df_cm = pd.DataFrame(confm, index=activity_names, columns=activity_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap="YlGnBu")
    out_fig = dataset + '_confusion_matrix.png'
    plt.savefig(os.path.join('result', out_fig))

    if(arg2 == 'include_novelty_exp') or (arg3 == 'include_novelty_exp') or (arg4 == 'include_novelty_exp'):
        novelty_detection_exp(model_hsa, model_vae, X_train, X_test, X_holdout)


if __name__ == "__main__":
    test_hsa_model(dataset, arg2, arg3, arg4)
