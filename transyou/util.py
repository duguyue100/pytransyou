"""Utility functions for TransYou package.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function

import os
from os.path import join
import numpy as np
from scipy.misc import imresize
import h5py
import cPickle as pickle

import transyou


def load_cifar_batch(folder_name, batch_fn):
    """Load a batch of CIFAR dataset.

    Assume the data is saved at $HOME/.transyou/res

    Parameters
    ----------
    batch_fn : string
        the filename of the CIFAR batch.
    folder_name : string
        the folder name of the CIFAR dataset.

    Returns
    -------
    data : numpy.ndarray
        the data of the batch.
    """
    batch_file = join(transyou.TRANSYOU_RES, folder_name, batch_fn)

    if not os.path.isfile(batch_file):
        raise ValueError("The file %s does not exit!" % (batch_file))

    fo = open(batch_file, "rb")
    data = pickle.load(fo)
    fo.close()

    data = data["data"]
    num_samples = data.shape[0]
    return data.reshape((num_samples, 3, 32, 32)).transpose((0, 2, 3, 1))


def load_stl10_batch(folder_name, batch_fn):
    """Load STL-10 batch.

    Parameters
    ----------
    folder_name : string
        the folder name of STL-10 dataset.
    batch_fn : string
        the file name of STL-10 bath data.

    Returns
    -------
    data : numpy.ndarray
        the data of the batch.
    """
    batch_file = join(transyou.TRANSYOU_RES, folder_name, batch_fn)

    if not os.path.isfile(batch_file):
        raise ValueError("The batch file %s does not exist." % (batch_file))

    fo = open(batch_file, "rb")
    data = np.fromfile(fo, dtype="uint8")
    data = data.reshape((-1, 3, 96, 96)).transpose((0, 3, 2, 1))

    data = list(data)
    for i in xrange(len(data)):
        data[i] = imresize(data[i], (32, 32))

    data = np.asarray(data, dtype="uint8")

    return data


def save_cifar10(db):
    """Save CIFAR-10 to HDF5 dataset.

    Parameters
    ----------
    db : h5py.File
        file object for HDF5 dataset.
    """
    folder_name = "cifar-10-batches-py"
    cifar10_ds = db.create_dataset("CIFAR10", (10000, 32, 32, 3),
                                   maxshape=(None, 32, 32, 3),
                                   dtype="uint8")

    for i in xrange(1, 6):
        fn = "data_batch_"+str(i)
        data = load_cifar_batch(folder_name, fn)
        cifar10_ds.resize(cifar10_ds.shape[0]+data.shape[0], axis=0)
        cifar10_ds[-data.shape[0]:, :, :, :] = data

    print ("[MESSAGE] CIFAR-10 dataset is saved.")
    print ("[MESSAGE] The dataset size is ", cifar10_ds.shape)


def save_cifar100(db):
    """Save CIFAR-100 to HDF5 dataset.

    Parameters
    ----------
    db : h5py.File
        file object for HDF5 dataset.
    """
    folder_name = "cifar-100-python"
    data = load_cifar_batch(folder_name, "train")

    cifar100_ds = db.create_dataset("CIFAR-100", data.shape,
                                    maxshape=(None, 32, 32, 3),
                                    dtype="uint8",
                                    data=data)

    data = load_cifar_batch(folder_name, "test")

    cifar100_ds.resize(cifar100_ds.shape[0]+data.shape[0], axis=0)
    cifar100_ds[-data.shape[0]:, :, :, :] = data

    print ("[MESSAGE] CIFAR-100 dataset is saved.")
    print ("[MESSAGE] The dataset size is ", cifar100_ds.shape)


def save_stl10(db):
    """Save STL-10 to HDF5 dataset.

    Parameters
    ----------
    db : h5py.File
        file object for HDF5 dataset.
    """
    folder_name = "stl10_binary"
    stl10_ds = db.create_dataset("STL-10", (10, 32, 32, 3),
                                 maxshape=(None, 32, 32, 3),
                                 dtype="uint8")

    data = load_stl10_batch(folder_name, "train_X.bin")
    stl10_ds.resize(data.shape[0], axis=0)
    stl10_ds[:, :, :, :] = data

    data = load_stl10_batch(folder_name, "test_X.bin")
    stl10_ds.resize(stl10_ds.shape[0]+data.shape[0], axis=0)
    stl10_ds[-data.shape[0]:, :, :, :] = data

    data = load_stl10_batch(folder_name, "unlabeled_X.bin")
    stl10_ds.resize(stl10_ds.shape[0]+data.shape[0], axis=0)
    stl10_ds[-data.shape[0]:, :, :, :] = data

    print ("[MESSAGE] CIFAR-100 dataset is saved.")
    print ("[MESSAGE] The dataset size is ", stl10_ds.shape)


def save_dataset(db_name):
    """Save CIFAR-10, CIFAR-100, STL-10 to HDF-10 format.

    Parameters
    ----------
    db_name : string
        the name of the database.
    """
    db_file = join(transyou.TRANSYOU_RES, db_name)
    if os.path.isfile(db_file):
        raise ValueError("The dataset %s exists." % (db_file))

    db_file = h5py.File(db_file, "w")

    # create CIFAR-10 dataset
    save_cifar10(db_file)

    # create CIFAR-100 dataset
    save_cifar100(db_file)

    # create STL-10 dataset
    save_stl10(db_file)

    db_file.close()
