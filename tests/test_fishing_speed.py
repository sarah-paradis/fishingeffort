import pandas as pd
import fishingeffort.fishingeffort as fe
import pandas.api.types as ptypes
import fishingeffort.show as fe_show
import os
import numpy as np


def test_fishing_speed():
    df = fe.read_df(file='AIS_data_test.csv', file_dir='tests')
    speed_parameters = fe.define_fishing_speed(df=df, speed_column='Sog', mean_trawl=2, mean_nav=10)
    assert isinstance(speed_parameters, pd.DataFrame)
    assert speed_parameters.shape == (2, 4)
    assert all(col in speed_parameters for col in ['mean', 'std_dev', 'range_lower', 'range_upper'])
    assert all(ptypes.is_numeric_dtype(speed_parameters[col]) for col in ['mean', 'std_dev',
                                                                          'range_lower', 'range_upper'])
    assert all(index in speed_parameters.index for index in ['trawling', 'navigating'])
    assert np.round(speed_parameters['mean']['trawling'], 2) == 2.55
    assert np.round(speed_parameters['std_dev']['trawling'], 2) == 0.78
    assert np.round(speed_parameters['range_lower']['trawling'], 2) == 1.02
    assert np.round(speed_parameters['range_upper']['trawling'], 2) == 4.08
    assert np.round(speed_parameters['mean']['navigating'], 2) == 8.83
    assert np.round(speed_parameters['std_dev']['navigating'], 2) == 2.03
    assert np.round(speed_parameters['range_lower']['navigating'], 2) == 4.86
    assert np.round(speed_parameters['range_upper']['navigating'], 2) == 12.80


def test_speed_filter():
    df = fe.read_df(file='AIS_data_test.csv', file_dir='tests')
    df_out = fe.filter_speeds(df=df, speed_column='Sog', min_speed=0, max_speed=20)
    assert df_out['Sog'].min() >= 0
    assert df_out['Sog'].max() <= 20


def test_plot_define_fishing_speed():
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    df = fe.read_df(file='AIS_data_test.csv', file_dir='tests')
    speed_parameters = fe.define_fishing_speed(df=df, speed_column='Sog', mean_trawl=2, mean_nav=10)
    fe_show.draw_histogram(data=df, df_speed_params=speed_parameters, dir_out='tests',
                           speed_unit='kn', fig_name='speed_hist', format_fig='jpg')
    assert os.path.exists(os.path.join('tests', 'speed_hist.jpg'))
