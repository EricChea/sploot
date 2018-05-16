"""Utilities to help manage memory.
"""

import numpy as np

NP_INTDTYPES = [np.int8, np.int16, np.int32, np.int64]
NP_INTDTYPES_EXP = [8, 16, 32, 64]


def minimize_intdtype(data):
    """
    data: np.array, one dimensional array
    """

    dtype = data.dtype
    ind = NP_INTDTYPES.index(dtype)
    max_val = abs(data).max()

    for i, exp in enumerate(NP_INTDTYPES_EXP[:ind]):

        if max_val <= 2**exp:
            dtype = np.dtype(NP_INTDTYPES[i])
            break

    return dtype


def minimize_dtypes(df):
    """Appends date attributes when the data contains unique data.  Convenience
    function to determine which data attributes are neccessary.

    Parameters
    ----------
    df: pd.DataFrame, Pandas DataFrame

    Returns
    -------
    dtypes dictionary of the best use of memory for the dtypes in each column
    if inplace is False, else None
    """

    dtypes = dict(df.dtypes.items())

    for col, dtype in dtypes.items():

        if dtype in (np.dtype(_dtype) for _dtype in NP_INTDTYPES):
            dtypes[col] = minimize_intdtype(data=df[col].values)

    return dtypes
