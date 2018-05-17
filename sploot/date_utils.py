
import numpy as np
import pandas as pd

from sploot.memory_utils import minimize_dtypes

def append_dateattrs(vector, inplace=False, datetime_props=(
    'year', 'month', 'weekofyear', 'dayofweek', 'dayofyear')):
    """Appends date attributes when the data contains unique data.  Convenience
    function to determine which data attributes are neccessary.

    Parameters
    ----------
    vector: 1D vector, data in np.datetime64 format
    inplace: (Optional) bool, True to modify the pd.DataFrame else a view of
        the results is returned. Default is False.
    datetime_props: (Optional) list, list of properties that pair up with
        Pandas Series datetime properties [1]. Default is ('year', 'month',
        'weekofyear', 'dayofweek', 'dayofyear').

        'year': the year of the datetime
        'month': the month as January=1, December=12
        'hour': the hours of the datetime
        'minute': the minute of the datetime
        'second': the seconds of the datetime
        'weekofyear': the week ordinal of the year
        'dayofweek': the day of the week with Monday=0, Sunday=6
        'dayofyear': the ordinal day of the year
        'quarter': the quarter of the date

        [1] https://pandas.pydata.org/pandas-docs/stable/api.html#time-series-related

    Return
    ------
    DataFrame with new columns in place.
    """

    func_map = {
        'year': lambda x: x.dt.year,
        'month': lambda x: x.dt.month,
        'hour': lambda x: x.dt.hour,
        'minute': lambda x: x.dt.minute,
        'second': lambda x: x.dt.second,
        'weekofyear': lambda x: x.dt.weekofyear,
        'dayofweek': lambda x: x.dt.dayofweek,
        'dayofyear': lambda x: x.dt.dayofyear,
        'quarter': lambda x: x.dt.quarter,
    }

    dateattrs = np.zeros((len(vector), len(datetime_props)))

    for i, dtprop in enumerate(datetime_props):
        dateattrs[:, i] = func_map[dtprop](vector)

    df = pd.DataFrame(
        data=dateattrs, columns=datetime_props, index=vector.index, dtype=np.int32
    )

    unique_indx = [i for i, val in enumerate(df.apply(lambda x: len(x.unique()) > 1)) if val]
    
    return df.iloc[:, unique_indx].astype(dtype=minimize_dtypes(df.iloc[:, unique_indx]))
