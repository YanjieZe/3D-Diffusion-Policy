import numpy as np
import pickle
import gzip


# This can do the compression.
def gload_data(file_path):
    file = gzip.GzipFile(file_path, 'rb')
    res = pickle.load(file)
    file.close()
    return res


def gsave_data(obj, file_path):
    file = gzip.GzipFile(file_path, 'wb')
    pickle.dump(obj, file, -1)
    file.close()


def save_data(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def load_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)