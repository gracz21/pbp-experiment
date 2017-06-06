import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import LabelEncoder


def read_all_data():
    path = r'./datasets/data'
    filenames = glob.glob(os.path.join(path, "*.csv"))
    data_list = []
    for file in filenames:
        df = pd.read_csv(file, sep=';', header=None)
        label_encoder = LabelEncoder()
        feature = label_encoder.fit_transform(df.iloc[:, -1])
        df[df.shape[1] - 1] = feature
        df = df.sample(frac=1).reset_index(drop=True)
        classes = len(set(df.iloc[:, -1]))
        train, valid, test = np.split(df, [round(len(df) * 2 / 3), round(len(df) * 5 / 6)])
        data_list.append({'train': train, 'valid': valid, 'test': test, 'classes': classes,
                          'file': os.path.basename(file)})
    return data_list
