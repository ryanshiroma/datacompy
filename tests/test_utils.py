from decimal import Decimal
import pytest
import pandas as pd
from pandas.util.testing import assert_series_equal
import numpy as np
import six

from datacompy import utils

def test_numeric_columns_equal_abs():
    data = '''a|b|expected
1|1|True
2|2.1|True
3|4|False
4|NULL|False
NULL|4|False
NULL|NULL|True'''
    dataframe = pd.read_csv(six.StringIO(data), sep='|')
    actual_out = utils.columns_equal(dataframe.a, dataframe.b, abs_tol=0.2)
    expect_out = dataframe['expected']
    assert_series_equal(expect_out, actual_out, check_names=False)

def test_numeric_columns_equal_rel():
    data = '''a|b|expected
1|1|True
2|2.1|True
3|4|False
4|NULL|False
NULL|4|False
NULL|NULL|True'''
    dataframe = pd.read_csv(six.StringIO(data), sep='|')
    actual_out = utils.columns_equal(dataframe.a, dataframe.b, rel_tol=0.2)
    expect_out = dataframe['expected']
    assert_series_equal(expect_out, actual_out, check_names=False)

def test_string_columns_equal():
    data = '''a|b|expected
Hi|Hi|True
Yo|Yo|True
Hey|Hey |False
résumé|resume|False
résumé|résumé|True
💩|💩|True
💩|🤔|False
 | |True
  | |False
datacompy|DataComPy|False
something||False
|something|False
||True'''
    dataframe = pd.read_csv(six.StringIO(data), sep='|')
    actual_out = utils.columns_equal(dataframe.a, dataframe.b, rel_tol=0.2)
    expect_out = dataframe['expected']
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_date_columns_equal():
    data = '''a|b|expected
2017-01-01|2017-01-01|True
2017-01-02|2017-01-02|True
2017-10-01|2017-10-10|False
2017-01-01||False
|2017-01-01|False
||True'''
    dataframe = pd.read_csv(six.StringIO(data), sep='|')
    #First compare just the strings
    actual_out = utils.columns_equal(dataframe.a, dataframe.b, rel_tol=0.2)
    expect_out = dataframe['expected']
    assert_series_equal(expect_out, actual_out, check_names=False)

    #Then compare converted to datetime objects
    dataframe['a'] = pd.to_datetime(dataframe['a'])
    dataframe['b'] = pd.to_datetime(dataframe['b'])
    actual_out = utils.columns_equal(dataframe.a, dataframe.b, rel_tol=0.2)
    expect_out = dataframe['expected']
    assert_series_equal(expect_out, actual_out, check_names=False)
    #and reverse
    actual_out_rev = utils.columns_equal(dataframe.b, dataframe.a, rel_tol=0.2)
    assert_series_equal(expect_out, actual_out_rev, check_names=False)



def test_date_columns_unequal():
    """I want datetime fields to match with dates stored as strings
    """
    dataframe = pd.DataFrame([
        {'a': '2017-01-01', 'b': '2017-01-02'},
        {'a': '2017-01-01'}
        ])
    dataframe['a_dt'] = pd.to_datetime(dataframe['a'])
    dataframe['b_dt'] = pd.to_datetime(dataframe['b'])
    assert utils.columns_equal(dataframe.a, dataframe.a_dt).all()
    assert utils.columns_equal(dataframe.b, dataframe.b_dt).all()
    assert utils.columns_equal(dataframe.a_dt, dataframe.a).all()
    assert utils.columns_equal(dataframe.b_dt, dataframe.b).all()
    assert not utils.columns_equal(dataframe.b_dt, dataframe.a).any()
    assert not utils.columns_equal(dataframe.a_dt, dataframe.b).any()
    assert not utils.columns_equal(dataframe.a, dataframe.b_dt).any()
    assert not utils.columns_equal(dataframe.b, dataframe.a_dt).any()


def test_bad_date_columns():
    """If strings can't be coerced into dates then it should be false for the
    whole column.
    """
    dataframe = pd.DataFrame([
        {'a': '2017-01-01', 'b': '2017-01-01'},
        {'a': '2017-01-01', 'b': '217-01-01'}
        ])
    dataframe['a_dt'] = pd.to_datetime(dataframe['a'])
    assert not utils.columns_equal(dataframe.a_dt, dataframe.b).any()


def test_rounded_date_columns():
    """If strings can't be coerced into dates then it should be false for the
    whole column.
    """
    dataframe = pd.DataFrame([
        {'a': '2017-01-01', 'b': '2017-01-01 00:00:00.000000', 'exp': True},
        {'a': '2017-01-01', 'b': '2017-01-01 00:00:00.123456', 'exp': False},
        {'a': '2017-01-01', 'b': '2017-01-01 00:00:01.000000', 'exp': False},
        {'a': '2017-01-01', 'b': '2017-01-01 00:00:00', 'exp': True}
        ])
    dataframe['a_dt'] = pd.to_datetime(dataframe['a'])
    actual = utils.columns_equal(dataframe.a_dt, dataframe.b)
    expected = dataframe['exp']
    assert_series_equal(actual, expected, check_names=False)


def test_decimal_float_columns_equal():
    dataframe = pd.DataFrame([
        {'a': Decimal('1'), 'b': 1, 'expected': True},
        {'a': Decimal('1.3'), 'b': 1.3, 'expected': True},
        {'a': Decimal('1.000003'), 'b': 1.000003, 'expected': True},
        {'a': Decimal('1.000000004'), 'b': 1.000000003, 'expected': False},
        {'a': Decimal('1.3'), 'b': 1.2, 'expected': False},
        {'a': np.nan, 'b': np.nan, 'expected': True},
        {'a': np.nan, 'b': 1, 'expected': False},
        {'a': Decimal('1'), 'b': np.nan, 'expected': False}
        ])
    actual_out = utils.columns_equal(dataframe.a, dataframe.b)
    expect_out = dataframe['expected']
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_decimal_float_columns_equal_rel():
    dataframe = pd.DataFrame([
        {'a': Decimal('1'), 'b': 1, 'expected': True},
        {'a': Decimal('1.3'), 'b': 1.3, 'expected': True},
        {'a': Decimal('1.000003'), 'b': 1.000003, 'expected': True},
        {'a': Decimal('1.000000004'), 'b': 1.000000003, 'expected': True},
        {'a': Decimal('1.3'), 'b': 1.2, 'expected': False},
        {'a': np.nan, 'b': np.nan, 'expected': True},
        {'a': np.nan, 'b': 1, 'expected': False},
        {'a': Decimal('1'), 'b': np.nan, 'expected': False}
        ])
    actual_out = utils.columns_equal(dataframe.a, dataframe.b, abs_tol=0.001)
    expect_out = dataframe['expected']
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_decimal_columns_equal():
    dataframe = pd.DataFrame([
        {'a': Decimal('1'), 'b': Decimal('1'), 'expected': True},
        {'a': Decimal('1.3'), 'b': Decimal('1.3'), 'expected': True},
        {'a': Decimal('1.000003'), 'b': Decimal('1.000003'), 'expected': True},
        {'a': Decimal('1.000000004'), 'b': Decimal('1.000000003'), 'expected': False},
        {'a': Decimal('1.3'), 'b': Decimal('1.2'), 'expected': False},
        {'a': np.nan, 'b': np.nan, 'expected': True},
        {'a': np.nan, 'b': Decimal('1'), 'expected': False},
        {'a': Decimal('1'), 'b': np.nan, 'expected': False}
        ])
    actual_out = utils.columns_equal(dataframe.a, dataframe.b)
    expect_out = dataframe['expected']
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_decimal_columns_equal_rel():
    dataframe = pd.DataFrame([
        {'a': Decimal('1'), 'b': Decimal('1'), 'expected': True},
        {'a': Decimal('1.3'), 'b': Decimal('1.3'), 'expected': True},
        {'a': Decimal('1.000003'), 'b': Decimal('1.000003'), 'expected': True},
        {'a': Decimal('1.000000004'), 'b': Decimal('1.000000003'), 'expected': True},
        {'a': Decimal('1.3'), 'b': Decimal('1.2'), 'expected': False},
        {'a': np.nan, 'b': np.nan, 'expected': True},
        {'a': np.nan, 'b': Decimal('1'), 'expected': False},
        {'a': Decimal('1'), 'b': np.nan, 'expected': False}
        ])
    actual_out = utils.columns_equal(dataframe.a, dataframe.b, abs_tol=0.001)
    expect_out = dataframe['expected']
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_infinity_and_beyond():
    dataframe = pd.DataFrame([
        {'a': np.inf, 'b': np.inf, 'expected': True},
        {'a': -np.inf, 'b': -np.inf, 'expected': True},
        {'a': -np.inf, 'b': np.inf, 'expected': False},
        {'a': np.inf, 'b': -np.inf, 'expected': False},
        {'a': 1, 'b': 1, 'expected': True},
        {'a': 1, 'b': 0, 'expected': False}
        ])
    actual_out = utils.columns_equal(dataframe.a, dataframe.b)
    expect_out = dataframe['expected']
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_mixed_column():
    dataframe = pd.DataFrame([
        {'a': 'hi', 'b': 'hi', 'expected': True},
        {'a': 1, 'b': 1, 'expected': True},
        {'a': np.inf, 'b': np.inf, 'expected': True},
        {'a': Decimal('1'), 'b': Decimal('1'), 'expected': True},
        {'a': 1, 'b': '1', 'expected': False},
        {'a': 1, 'b': 'yo', 'expected': False}
        ])
    actual_out = utils.columns_equal(dataframe.a, dataframe.b)
    expect_out = dataframe['expected']
    assert_series_equal(expect_out, actual_out, check_names=False)


def test_temp_column_name():
    df1 = pd.DataFrame([{'a': 'hi', 'b': 2}, {'a': 'bye', 'b': 2}])
    df2 = pd.DataFrame([{'a': 'hi', 'b': 2}, {'a': 'bye', 'b': 2}, {'a': 'back fo mo', 'b': 3}])
    actual = utils.temp_column_name(df1, df2)
    assert actual == '_temp_0'

def test_temp_column_name_one_has():
    df1 = pd.DataFrame([{'_temp_0': 'hi', 'b': 2}, {'_temp_0': 'bye', 'b': 2}])
    df2 = pd.DataFrame([{'a': 'hi', 'b': 2}, {'a': 'bye', 'b': 2}, {'a': 'back fo mo', 'b': 3}])
    actual = utils.temp_column_name(df1, df2)
    assert actual == '_temp_1'

def test_temp_column_name_both_have():
    df1 = pd.DataFrame([{'_temp_0': 'hi', 'b': 2}, {'_temp_0': 'bye', 'b': 2}])
    df2 = pd.DataFrame([{'_temp_0': 'hi', 'b': 2}, {'_temp_0': 'bye', 'b': 2}, {'a': 'back fo mo', 'b': 3}])
    actual = utils.temp_column_name(df1, df2)
    assert actual == '_temp_1'

def test_temp_column_name_both_have():
    df1 = pd.DataFrame([{'_temp_0': 'hi', 'b': 2}, {'_temp_0': 'bye', 'b': 2}])
    df2 = pd.DataFrame([{'_temp_0': 'hi', 'b': 2}, {'_temp_1': 'bye', 'b': 2}, {'a': 'back fo mo', 'b': 3}])
    actual = utils.temp_column_name(df1, df2)
    assert actual == '_temp_2'

def test_temp_column_name_one_already():
    df1 = pd.DataFrame([{'_temp_1': 'hi', 'b': 2}, {'_temp_1': 'bye', 'b': 2}])
    df2 = pd.DataFrame([{'_temp_1': 'hi', 'b': 2}, {'_temp_1': 'bye', 'b': 2}, {'a': 'back fo mo', 'b': 3}])
    actual = utils.temp_column_name(df1, df2)
    assert actual == '_temp_0'
