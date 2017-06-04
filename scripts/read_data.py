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
        df = pd.read_csv(file, sep=';', header=None, na_values='?')
        matrix = df.as_matrix()
        features = []
        for i in range(0, matrix.shape[1]):
            column = matrix[:, i]
            if any(isinstance(x, str) for x in column):
                column[pd.isnull(column)] = 'NaN'
            label_encoder = LabelEncoder()
            feature = label_encoder.fit_transform(column)
            features.append(feature)
        encoded = np.array(features)
        encoded = encoded.transpose()
        train, valid, test = np.split(encoded, [round(len(encoded) * 2 / 3), round(len(encoded) * 5 / 6)])
        classes = set(encoded[:, -1])
        data_list.append({'train': train, 'valid': valid, 'test': test, 'classes': len(classes), 'file': file})
    return data_list
