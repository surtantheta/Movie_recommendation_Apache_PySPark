
small_dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'

import os

datasets_path = os.path.abspath(os.path.join('datasets'))

small_dataset_path = os.path.join(datasets_path, 'ml-latest-small.zip')
print(small_dataset_path)
import urllib.request

small_f = urllib.request.urlretrieve (small_dataset_url, small_dataset_path)

import zipfile

with zipfile.ZipFile(small_dataset_path, "r") as z:
    z.extractall(datasets_path)
