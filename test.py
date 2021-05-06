import argparse
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

def parse_args():
    parser = argparse.ArgumentParser(description='Hierarchical Self Attention Based Autoencoder for Open-Set Human Activity Recognition')
    parser.add_argument('--dataset', '-D', default='daphnet', help='Dataset name- options: opp, pamap2, daphhet, skoda')
    parser.add_argument('--use_pretrained', action='store_true', default=False, help='Use pretrained model')
    parser.add_argument('--include_openset_exp', action='store_true', default=False, help='Perform Openset Recognition')
    parser.add_argument('--save_weights', action='store_true', default=False, help='Save model parameters')
    return parser.parse_args()


def test_hsa_model(dataset, use_pretrained=False, include_openset_exp=False, save_weights=False):
    print(tabulate([['Hierarchical Self Attention Based Autoencoder for Open-Set Human Activity Recognition']], [
    ], tablefmt="fancy_grid"))
    print('[PREPROCESSING AND LOADING DATA] ...')
    if use_pretrained and include_openset_exp:
        if (arg2 == '') or (arg3 == 'include_novelty_exp') or (arg4 == 'include_novelty_exp'):
            print('Novelty experiment using pretrained weights is not currently supported. Please conduct novelty experiment with model training.')
            return
        else:
            (X_train, y_train), (X_test, y_test) = get_train_test_data(
                dataset=dataset, holdout=False)
            if os.path.exists(os.path.join('saved_models', dataset)):
                model_hsa = train_model(
                    dataset, (X_train, y_train), train_hsa=False)
                model_hsa.load_weights(os.path.join(
                    'saved_models', dataset, dataset)).expect_partial()
            else:
                print('Pretrained weights not available, starting training')
                model_hsa = train_model(
                    dataset, (X_train, y_train), train_hsa=True)
            # if
            # model_hsa.load
    else:
        if include_openset_exp:
            (X_train, y_train),  (X_test, y_test), (X_holdout,
                                                    y_holdout) = get_train_test_data(dataset=dataset, holdout=True)
            model_hsa, model_vae = train_model(
                dataset, (X_train, y_train), train_vae=True)
        else:
            (X_train, y_train, y_train_mid), (X_val, y_val, y_val_mid), (X_test, y_test, y_test_mid) = get_train_test_data(
                dataset=dataset, holdout=False)
            if X_val is None:
                val_data = None
            else:
                val_data = (X_val, y_val, y_val_mid)
            model_hsa = train_model(dataset, (X_train, y_train, y_train_mid),  val_data=val_data)

    if save_weights:
        if not os.path.exists(os.path.join('saved_models', dataset)):
            os.mkdir(os.path.join('saved_models', dataset))
        model_hsa.save_weights(os.path.join('saved_models', dataset, dataset))

    pred_mid, pred_sess = model_hsa.predict(
        X_test, batch_size=hyperparameters['test']['batch_size'])

    activity_map = json.load(
        open(os.path.join('data', 'activity_maps', dataset + '_activity.json')))

    if include_openset_exp:
        metadata = yaml.load(metadata_file, Loader=yaml.FullLoader)[
            dataset + '_preprocess']
        NOVEL_CLASSES = metadata['NOVEL_CLASSES']
        activity_map, novel_map = get_activity_dict(
            activity_map, NOVEL_CLASSES)
        print('\nNOVEL / UNSEEN ACTIVITIES: ', novel_map)
        print()

    activity_names = list(activity_map.values())
    print("Window level:")
    print(classification_report(np.argmax(y_test_mid.reshape(-1, 2), axis=1), np.argmax(pred_mid.reshape(-1, 2), axis=1), labels=range(len(activity_names)), target_names=activity_names, zero_division=1))
    confm = confusion_matrix(np.argmax(y_test_mid.reshape(-1, y_test.shape[1]), axis=1), np.argmax(pred_mid.reshape(-1, y_test.shape[1]), axis=1), labels=range(len(activity_names)))
    print(confm)

    print("Session level:")
    print(classification_report(np.argmax(y_test, axis=1), np.argmax(pred_sess, axis=1),
                                labels=range(len(activity_names)), target_names=activity_names, zero_division=1))
    # out_res = open(os.path.join('result', dataset + '_classification_report.txt'))
    # print(classification_report(np.argmax(y_test, axis=1), np.argmax(pred, axis=1),labels = range(len(activity_names)), target_names=activity_names, zero_division=1), file=out_res)
    # out_res.close()

    confm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(
        pred_sess, axis=1), labels=range(len(activity_names)))
    print(confm)

    df_cm = pd.DataFrame(confm, index=activity_names, columns=activity_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap="YlGnBu")
    out_fig = dataset + '_confusion_matrix.png'
    plt.savefig(os.path.join('result', out_fig))

    if include_openset_exp:
        novelty_detection_exp(model_hsa, model_vae, X_train, X_test, X_holdout)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    test_hsa_model(args.dataset, use_pretrained=args.use_pretrained, save_weights=args.save_weights, include_openset_exp=args.include_openset_exp)
