import copy
import csv
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.preprocessing import StandardScaler


class data_reader:
    def __init__(self, train_test_files, use_columns, output_file_name, labelToId):
        self.data = self.readPamap2(train_test_files, use_columns, labelToId)
        self.save_data(output_file_name)

    def save_data(self, output_file_name):
        f = h5py.File(output_file_name)
        for key in self.data:
            f.create_group(key)
            for field in self.data[key]:
                f[key].create_dataset(field, data=self.data[key][field])
        f.close()
        print('Done.')

    @property
    def train(self):
        return self.data['train']

    @property
    def test(self):
        return self.data['test']

    def readPamap2(self,train_test_files,use_columns, labelToId):
        files = train_test_files
        cols = use_columns
        data = {dataset: self.readPamap2Files(files[dataset], cols, labelToId)
                for dataset in ('train', 'test', 'validation')}
        return data

    def readPamap2Files(self, filelist, cols, labelToId):
        data = []
        labels = []
        base_path = os.path.join('data','raw', 'pamap2', 'PAMAP2_Dataset', 'Protocol')
        assert os.path.exists(base_path), "Please download the dataset first using the script"
        for i, filename in enumerate(filelist):
            # print('Reading file %d of %d' % (i+1, len(filelist)))
            with open(os.path.join(base_path, filename), 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                for line in reader:
                    elem = []
                    if line[1] == "0":
                        continue
                    for ind in cols:
                        elem.append(line[ind])
                    if sum([x == 'NaN' for x in elem]) < 9:
                        data.append([float(x) / 100 for x in elem[:-1]])
                        labels.append(labelToId[elem[0]])
                        if elem[0] == 1:
                            print(labelToId[elem[0]])
        
        return {'inputs': np.asarray(data), 'targets': np.asarray(labels, dtype=int)+1}


def read_dataset_pamap2(train_test_files, use_columns, output_file_name, label_to_id):
    # print('Reading pamap2')
    dr = data_reader(train_test_files, use_columns, output_file_name, label_to_id)

def preprocess_pamap2(file, downsample=False, print_debug =False):
    path = os.path.join(file)
    f = h5py.File(path, 'r')

    train_x = f.get('train').get('inputs')[()]
    train_y = f.get('train').get('targets')[()]

    val_x = f.get('validation').get('inputs')[()]
    val_y = f.get('validation').get('targets')[()]

    test_x = f.get('test').get('inputs')[()]
    test_y = f.get('test').get('targets')[()]

    if print_debug:
        print("x_train shape = ", train_x.shape)
        print("y_train shape =", train_y.shape)
        print("x_val shape = ", val_x.shape)
        print("y_val shape =", val_y.shape)
        print("x_test shape =" ,test_x.shape)
        print("y_test shape =",test_y.shape)

    if downsample:
        train_x = train_x[::3,:]
        train_y = train_y[::3]
        val_x = val_x[::3,:]
        val_y = val_y[::3]
        test_x = test_x[::3,:]
        test_y = test_y[::3]

    if print_debug:
        print("x_train shape = ", train_x.shape)
        print("y_train shape =", train_y.shape)
        print("x_val shape = ", val_x.shape)
        print("y_val shape =", val_y.shape)
        print("x_test shape =" ,test_x.shape)
        print("y_test shape =",test_y.shape)

    train_x = np.nan_to_num(train_x)
    val_x = np.nan_to_num(val_x)
    test_x = np.nan_to_num(test_x)

    return (train_x, train_y), (val_x, val_y), (test_x, test_y)
