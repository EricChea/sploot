"""
"""
import pandas as pd

def count_nans(df):

    return {col: pd.isnull(df[col]).sum() for col in df.columns}