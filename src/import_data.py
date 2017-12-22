import os
import tarfile
import zipfile
from six.moves import urllib

SRC = os.path.abspath(os.path.dirname('__file__'))
DATA_ROOT = os.path.abspath(os.path.relpath('../data/', SRC))

def fetch_tgz_data(url, data_name, data_dir='', data_root=DATA_ROOT):
    data_full_dir = os.path.join(data_root, data_dir)
    if not os.path.isdir(data_full_dir):
        os.makedirs(data_full_dir)
    tgz_path = os.path.join(data_full_dir, data_name)
    urllib.request.urlretrieve(url, tgz_path)
    tgz_file = tarfile.open(tgz_path)
    tgz_file.extractall(path=data_full_dir)
    tgz_file.close()
    os.remove(tgz_path)

def fetch_zip_data(url, data_name, data_dir='', data_root=DATA_ROOT):
    data_full_dir = os.path.join(data_root, data_dir)
    if not os.path.isdir(data_full_dir):
        os.makedirs(data_full_dir)
    zip_path = os.path.join(data_full_dir, data_name)
    urllib.request.urlretrieve(url, zip_path)
    zip_file = zipfile.ZipFile(zip_path, 'r')
    zip_file.extractall(path=data_full_dir)
    zip_file.close()
    os.remove(zip_path)
