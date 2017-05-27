import pandas as pd
import glob
import os


def read_all_data():
    path = r'./data'
    filenames = glob.glob(os.path.join(path, "*.csv"))
    data_list = []
    for file in filenames:
        df = pd.read_csv(file, index_col=None, header=0)
        data_list.append(df)
    return data_list
