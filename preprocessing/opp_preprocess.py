import csv
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.preprocessing import StandardScaler


def create_windowed_dataset_opp(df, features, class_label, MID_LABEL_COL, LOCO_LABEL_COL, window_size=24, stride = 12):
    X = df[features].values
    y = df[class_label].values
    segments = []
    labels = []
    seg_start= 0
    seg_end = window_size
    mid_labels = []
    loco_labels = []
    while seg_end <= len(X):
        if len(np.unique(y[seg_start:seg_end])) == 1:
            segments.append(X[seg_start:seg_end])
            labels.append(y[seg_start]) 
            mid_labels.append(df[str(MID_LABEL_COL)].values[seg_start:seg_end])
            loco_labels.append(df[str(LOCO_LABEL_COL)].values[seg_start:seg_end])

            seg_start += stride
            seg_end = seg_start + window_size

        else:
            current_label = y[seg_start]
            for i in range(seg_start, seg_end):
                if y[i] != current_label:
                    seg_start = i
                    seg_end = seg_start + window_size
                    break

    return np.asarray(segments).astype(np.float32), np.asarray(labels), mid_labels, loco_labels


def readOpportunityFiles(filelist, cols, mid_label_to_id, hi_label_to_id, loco_label_to_id):
    data = []
    mid_labels = []
    hi_labels = []
    loco_labels = []
    subject_mapping = []
    base_path = os.path.join('data','raw', 'opp', 'OpportunityUCIDataset', 'dataset')
    assert os.path.exists(base_path), "Please download the dataset first using the script"

    for i, filename in enumerate(filelist):
        with open(os.path.join(base_path, filename), 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            subject_info = [int(filename[1]), filename[6:7]]
            for line in reader:
                elem = []
                for ind in cols:
                    elem.append(line[ind])
                if sum([x != 'NaN' for x in elem]) > 40:
                    data.append([float(x) for x in elem[:-3]])
                    mid_labels.append(mid_label_to_id[elem[-1]])
                    hi_labels.append(hi_label_to_id[elem[-2]])
                    loco_labels.append(loco_label_to_id[elem[-3]])
                    subject_mapping.append(subject_info)

    return np.asarray(data), np.asarray(mid_labels, dtype=int), np.asarray(hi_labels, dtype=int), np.asarray(loco_labels, dtype=int), np.asarray(subject_mapping)


def prepare_opp_data():
    metadata_file = open('configs/metadata.yaml', mode='r')
    metadata = yaml.load(metadata_file, Loader=yaml.FullLoader)['opp_preprocess']
    file_list_nodrill = metadata['file_list']
    
    mid_label_to_id = metadata['mid_label_to_id']
    hi_label_to_id = metadata['hi_label_to_id']
    loco_label_to_id = metadata['loco_label_to_id']

    cols = metadata['columns_list']

    selected_cols = np.asarray(cols)-1
    
    data, mid_labels, hi_labels, loco_labels, subject_mapping = readOpportunityFiles(file_list_nodrill, selected_cols, mid_label_to_id, hi_label_to_id, loco_label_to_id)
    
    shp = data.shape[0]
    combined = pd.DataFrame(np.hstack((data.astype(np.float32), loco_labels.reshape((shp, 1)), mid_labels.reshape((shp, 1)), hi_labels.reshape((shp, 1)), subject_mapping)))
    combined.to_csv(os.path.join('data', 'processed', 'clean_opp.csv'), index=False)
