import pandas as pd
import numpy as np
import glob
import os


def read_all_data():
    path = r'./data'
    filenames = glob.glob(os.path.join(path, "*.csv"))
    data_list = []
    for file in filenames:
        df = pd.read_csv(file, index_col=None, header=0)
        train, valid, test = np.split(df, [round(len(df) * 2 / 3), round(len(df) * 5 / 6)])
        data_list.append({'train': train, 'valid': valid, 'test': test})
    return data_list
