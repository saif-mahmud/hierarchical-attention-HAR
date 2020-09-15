import numpy as np
import pandas as pd

def create_windowed_dataset(df, features, class_label, window_size, stride):
    X = df[features].values
    y = df[class_label].values
    segments = []
    labels = []

    seg_start = 0
    seg_end = window_size
    while seg_end <= len(X):
        if len(np.unique(y[seg_start:seg_end])) == 1:  # valid-window
            segments.append(X[seg_start:seg_end])
            labels.append(y[seg_start])  # doesn't matter all same labels

            seg_start += stride
            seg_end = seg_start + window_size

        else:  # mixed- discard last few samples, move start to the new activity
            current_label = y[seg_start]
            for i in range(seg_start, seg_end):
                if y[i] != current_label:
                    seg_start = i
                    seg_end = seg_start + window_size
                    break

    return np.asarray(segments).astype(np.float32), np.asarray(labels)
