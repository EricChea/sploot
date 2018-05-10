"""Convenience functions for storing files.

The training data set refers to data that is used to fit a model.
The validation data set refers to data that is used to tune a model.
The test data set refers to the holdout data that is used to score the performance of a model.
"""

from sklearn.model_selection import train_test_split
import numpy as np

import h5py


def save_dataset(data, filepath, dataset_indx=None):
    """Standardize the storage of data.

    Parameters
    ----------
    data: np.array, numpy array.
    filepath: str, path to h5 file.
    dataset_indx: tuple or list, references to the indices in data for subsets of data. For example, ([0,2,3], [1]) indicates to use indices 0, 2, 3 for training and validation, and index 1 for testing. If None then a random 80/20 split is performed. Default is None.


    """

    if dataset_indx is None:
        t
        train_test_split()


def load_dataset(train_filepath, test_filepath):
    """Loads data from h5 storage.

    Parameters
    ----------
    filepath: str, path to h5 file.

    Returns
    -------
    X_train: np.array, (m, l, w, c) shaped array of features for training.
    y_train: np.array, (m x 1) shaped array of labels for X_train.
    X_test: np.array, (m, l, w, c) shaped array of features for testing.
    y_test: np.array, (m x 1) shaped array of labels for X_test.
    classes: list, list of all classes to identify the presence of.

    Notes
    -----
    Shorthand abbreviations:
    m = number of samples
    l = length of the image
    w = width of the image
    c = number of channels
    """

    train_dataset = h5py.File(filepath, "r")
    X_train = np.array(train_dataset["X_train"][:])
    y_train = np.array(train_dataset["y_train"][:])

    test_dataset = h5py.File(test_filepath, "r")
    X_test = np.array(test_dataset["X_test"][:])
    y_test = np.array(test_dataset["y_test"][:])

    classes = np.array(test_dataset["classes"][:])
    
    return X_train, y_train, X_test, y_test, classes


def reshape_image(img_path, image_reshape_size, array=True):
    """Reshapes an image to defined image size. Useful function for standardizing images.

    Parameters
    ----------
    img_path: str, path to an image file.
    image_reshape_size: tuple, (length, width) of an image.
    array: bool, true to convert image into a numpy array else return an RGB image.

    Returns
    -------
    np.array or rgb image
    """
    im = Image.open(img_path)
    im = im.convert('RGB')
    im = im.resize(image_reshape_size, Image.BICUBIC)

    if array:
        im = np.array(im)
        assert len(im.shape) == 3
        assert im.shape[0] == image_reshape_size[0]
        assert im.shape[1] == image_reshape_size[1]
        assert im.shape[2] == 3

    return im