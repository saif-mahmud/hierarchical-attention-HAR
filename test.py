import os
import sys
import yaml
import json

from preprocessing.utils import get_train_test_data
from train import train_model

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


hparam_file = open('configs/hyperparameters.yaml', mode='r')
hyperparameters = yaml.load(hparam_file, Loader=yaml.FullLoader)

dataset = str(sys.argv[1])


def test_hsa_model(dataset, model=None):
    if model is None:
        (X_train, y_train), (X_test, y_test) = get_train_test_data(dataset=dataset)
        model = train_model(dataset, (X_train, y_train))
    pred = model.predict(X_test, batch_size=hyperparameters['test']['batch_size'])
    activity_map = json.load(open(os.path.join('data', 'activity_maps', dataset+'_activity.json')))
    activity_names = list(activity_map.values())
    print(classification_report(np.argmax(y_test, axis=1), np.argmax(pred, axis=1),labels = range(len(activity_names)), target_names=activity_names, zero_division=1))
    confm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(pred, axis=1), labels = range(len(activity_names)))
    df_cm = pd.DataFrame(confm, index=activity_names, columns=activity_names)
    plt.figure(figsize = (10,8))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap="YlGnBu")
    plt.show()


if __name__ == "__main__":
    print('hello')
    test_hsa_model(dataset)