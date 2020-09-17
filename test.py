import os
import sys
import yaml
import json
import warnings

from preprocessing.utils import get_train_test_data
from train import train_model
from experiments.novelty_detection import novelty_detection_exp

import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore")
# tf.get_logger().setLevel('INFO')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


hparam_file = open('configs/hyperparameters.yaml', mode='r')
hyperparameters = yaml.load(hparam_file, Loader=yaml.FullLoader)

dataset = str(sys.argv[1])
try:
    pretrained = str(sys.argv[2])
except IndexError:
    pretrained = None

try:
    novelty = str(sys.argv[3])
except IndexError:
    novelty = None


def test_hsa_model(dataset, pretrained='no', novelty='no'):
    print('\nPREPROCESSING AND LOADING DATA:')
    if pretrained != 'use_pretrained':
        if novelty == 'include_novelty_exp':
            (X_train, y_train),  (X_test, y_test), (X_holdout,
                                                    y_holdout) = get_train_test_data(dataset=dataset, holdout=True)
            model_hsa, model_vae = train_model(
                dataset, (X_train, y_train), train_vae=True)
        else:
            (X_train, y_train), (X_test, y_test) = get_train_test_data(
                dataset=dataset, holdout=False)
            model_hsa = train_model(dataset, (X_train, y_train))

    pred = model_hsa.predict(
        X_test, batch_size=hyperparameters['test']['batch_size'])

    activity_map = json.load(
        open(os.path.join('data', 'activity_maps', dataset+'_activity.json')))
    activity_names = list(activity_map.values())

    print(classification_report(np.argmax(y_test, axis=1), np.argmax(pred, axis=1),
                                labels=range(len(activity_names)), target_names=activity_names, zero_division=1))
    # out_res = os.path.join('result', dataset + '_classification_report.txt')
    # print(classification_report(np.argmax(y_test, axis=1), np.argmax(pred, axis=1),labels = range(len(activity_names)), target_names=activity_names, zero_division=1), file=out_res)

    confm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(
        pred, axis=1), labels=range(len(activity_names)))
    print(confm)

    df_cm = pd.DataFrame(confm, index=activity_names, columns=activity_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap="YlGnBu")
    out_fig = dataset + '_confusion_matrix.png'
    plt.savefig(os.path.join('result', out_fig))

    if novelty == 'include_novelty_exp':
        novelty_detection_exp(model_hsa, model_vae, X_train, X_test, X_holdout)
    else:
        pass


if __name__ == "__main__":
    test_hsa_model(dataset, pretrained, novelty)
