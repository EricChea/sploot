"""Utilities for transforming data.
"""

def explode_field(data, field):
    """Stacks a cell that represents multiple values (list)

    Parameters
    ----------
    data: pd.DataFrame, data containing the field of interest
    field: str, the column name containing a list.
    """

    kwargs = {col: np.repeat(data[col].values, data[field].str.len())
              for col in data.columns if col != field}
    kwargs.update({field: np.concatenate(data[field].values)})
    return pd.DataFrame(kwargs)
