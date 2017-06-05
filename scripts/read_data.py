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
        classes = set(df.as_matrix()[:, -1])
        train, valid, test = np.split(df, [round(len(df) * 2 / 3), round(len(df) * 5 / 6)])
        data_list.append({'train': train, 'valid': valid, 'test': test, 'classes': len(classes), 'file': file})
    return data_list
