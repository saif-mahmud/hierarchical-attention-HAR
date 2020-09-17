import os
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from tabulate import tabulate


def kl_div(z_mean, z_log_var):
    kl_loss = -0.5 * \
        tf.math.reduce_mean(
            (z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1), axis=1)
    return kl_loss.numpy()


def novelty_eval_kl(train_kl, test_kl, holdout_kl, mult=0.25):
    threshold = np.mean(train_kl) - (mult * np.std(train_kl))

    y_holdout = (holdout_kl > threshold).astype(int)
    y_true = np.ones(shape=holdout_kl.shape, dtype=int)

    y_test_n = (test_kl > threshold).astype(int)
    _y_true = np.zeros(shape=test_kl.shape, dtype=int)

    y_holdout = np.append(y_holdout, y_test_n)
    y_true = np.append(y_true, _y_true)

#     return accuracy_score(y_true, y_holdout), f1_score(y_true, y_holdout, average='macro')

    print(classification_report(y_true, y_holdout,
                                labels=[0, 1], target_names=['KNOWN', 'NOVEL']))


def novelty_eval_reconstrunction(train_rec_loss, test_rec_loss, novel_rec_loss, mult=0.25):
    threshold = np.mean(train_rec_loss) - (mult * np.std(train_rec_loss))

    y_holdout = (novel_rec_loss > threshold).astype(int)
    y_true = np.ones(shape=novel_rec_loss.shape, dtype=int)

    y_test_n = (test_rec_loss > threshold).astype(int)
    _y_true = np.zeros(shape=test_rec_loss.shape, dtype=int)

    y_holdout = np.append(y_holdout, y_test_n)
    y_true = np.append(y_true, _y_true)

    return accuracy_score(y_true, y_holdout), f1_score(y_true, y_holdout, average='macro'),

#     print(classification_report(y_true, y_holdout, labels=[0, 1], target_names=['KNOWN', 'NOVEL']))


def hparam_search(train_rec_loss, test_rec_loss, novel_rec_loss, plot=False):
    table = list()
    thresh_vals = list(np.arange(0.0, 1.01, 0.01))

    idx = 0

    for m_val in thresh_vals:
        result = novelty_eval_reconstrunction(
            train_rec_loss, test_rec_loss, novel_rec_loss, mult=m_val)
        table.append([idx, m_val, result[0], result[1]])

        idx = idx + 1

    table = np.array(table)

    if plot:
        sns.lineplot(table[:, 1], table[:, 2], label='Accuracy')
        sns.lineplot(table[:, 1], table[:, 3], label='Macro F1')
        plt.xlabel('Hyperparameter Value')
        plt.title('Novelty Detection Experiement')
        plt.show()

    top_acc = np.array(pd.Series(table[:, 2]).nlargest().index)
    top_f1 = np.array(pd.Series(table[:, 3]).nlargest().index)

    # print('Index with Top Accuracy and Macro F1 : ', top_acc, top_f1)
    # print('Most Important Index : ', np.intersect1d(top_acc, top_f1))

    print('HYPERPARAMETER SEARCH FOR NOVELTY DETECTION THRESHOLD:')
    print(tabulate(table, headers=[
          'Index', 'Std. Multiplier Value', 'Accuracy', 'Macro F1'], tablefmt="grid"))


def novelty_detection_exp(hsa_model, hsa_vae, X_train, X_test, X_holdout):
    hierarchical_model = tf.keras.Model(inputs=hsa_model.input, outputs=hsa_model.get_layer(
        'combined_sensor_self_attention_1').output, name='hierarchical_encoder')
    hierarchical_model.trainable = False

    z_mean, z_log_var, z = hsa_vae.encoder(
        hierarchical_model.predict(X_holdout)[0])

    train_rec_loss = tf.keras.losses.mean_squared_error(
        hierarchical_model.predict(X_train)[0], hsa_vae.predict(X_train)).numpy()
    test_rec_loss = tf.keras.losses.mean_squared_error(
        hierarchical_model.predict(X_test)[0], hsa_vae.predict(X_test)).numpy()
    novel_rec_loss = tf.keras.losses.mean_squared_error(
        hierarchical_model.predict(X_holdout)[0], hsa_vae.predict(X_holdout)).numpy()

    hparam_search(train_rec_loss, test_rec_loss, novel_rec_loss)
