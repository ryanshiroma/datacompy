# -*- coding: utf-8 -*-
#
# Copyright 2018 Capital One Services, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utilities for datacompy
"""

import os
import logging
import pandas as pd
import numpy as np

LOG = logging.getLogger(__name__)

def render(filename, **fields):
    """Renders out an individual template.  This basically just reads in a
    template file, and applies ``.format()`` on the fields.

    Parameters
    ----------
    filename : str
        The file that contains the template.  Will automagically prepend the
        templates directory before opening
    fields : dict
        Fields to be rendered out in the template

    Returns
    -------
    str
        The fully rendered out file.
    """
    this_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(this_dir, 'templates', filename)) as file_open:
        return file_open.read().format(**fields)


def columns_equal(col_1, col_2, rel_tol=0, abs_tol=0):
    """Compares two columns from a dataframe, returning a True/False series,
    with the same index as column 1.

    - Two nulls (np.nan) will evaluate to True.
    - A null and a non-null value will evaluate to False.
    - Numeric values will use the relative and absolute tolerances.
    - Decimal values (decimal.Decimal) will attempt to be converted to floats
      before comparing
    - Non-numeric values (i.e. where np.isclose can't be used) will just
      trigger True on two nulls or exact matches.

    Parameters
    ----------
    col_1 : Pandas.Series
        The first column to look at
    col_2 : Pandas.Series
        The second column
    rel_tol : float, optional
        Relative tolerance
    abs_tol : float, optional
        Absolute tolerance

    Returns
    -------
    pandas.Series
        A series of Boolean values.  True == the values match, False == the
        values don't match.
    """
    try:
        compare = pd.Series(np.isclose(
            col_1,
            col_2,
            rtol=rel_tol,
            atol=abs_tol,
            equal_nan=True))
    except TypeError:
        try:
            compare = pd.Series(np.isclose(
                col_1.astype(float),
                col_2.astype(float),
                rtol=rel_tol,
                atol=abs_tol,
                equal_nan=True))
        except (ValueError, TypeError):
            try:
                if set([col_1.dtype.kind, col_2.dtype.kind]) == set(['M','O']):
                    compare = compare_string_and_date_columns(col_1, col_2)
                else:
                    compare = pd.Series((col_1 == col_2)\
                        | (col_1.isnull() & col_2.isnull()))
            except:
                #Blanket exception should just return all False
                compare = pd.Series(False, index=col_1.index)
    compare.index = col_1.index
    return compare


def compare_string_and_date_columns(col_1, col_2):
    """Compare a string column and date column, value-wise.  This tries to
    convert a string column to a date column and compare that way.

    Parameters
    ----------
    col_1 : Pandas.Series
        The first column to look at
    col_2 : Pandas.Series
        The second column

    Returns
    -------
    pandas.Series
        A series of Boolean values.  True == the values match, False == the
        values don't match.
    """
    if col_1.dtype.kind == 'O':
        obj_column = col_1
        date_column = col_2
    else:
        obj_column = col_2
        date_column = col_1

    try:
        return pd.Series(
            (pd.to_datetime(obj_column) == date_column) \
            | (obj_column.isnull() & date_column.isnull()))
    except:
        return pd.Series(False, index=col_1.index)


def get_merged_columns(original_df, merged_df, suffix):
    """Gets the columns from an original dataframe, in the new merged dataframe

    Parameters
    ----------
    original_df : Pandas.DataFrame
        The original, pre-merge dataframe
    merged_df : Pandas.DataFrame
        Post-merge with another dataframe, with suffixes added in.
    suffix : str
        What suffix was used to distinguish when the original dataframe was
        overlapping with the other merged dataframe.
    """
    columns = []
    for col in original_df.columns:
        if col in merged_df.columns:
            columns.append(col)
        elif col + suffix in merged_df.columns:
            columns.append(col + suffix)
        else:
            raise ValueError('Column not found: %s', col)
    return columns


def temp_column_name(*dataframes):
    """Gets a temp column name that isn't included in columns of any dataframes

    Parameters
    ----------
    dataframes : list of Pandas.DataFrame
        The DataFrames to create a temporary column name for

    Returns
    -------
    str
        String column name that looks like '_temp_x' for some integer x
    """
    i = 0
    while True:
        temp_column = '_temp_{}'.format(i)
        unique = True
        for dataframe in dataframes:
            if temp_column in dataframe.columns:
                i += 1
                unique = False
        if unique:
            return temp_column
