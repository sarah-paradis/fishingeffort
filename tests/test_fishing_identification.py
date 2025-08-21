import fishingeffort.fishingeffort as fe
import fishingeffort.show as fe_show
import pandas.api.types as ptypes
import os


def test_identify_fishing_plot():
    df = fe.read_df(file='AIS_data_test.csv', file_dir='tests')
    df_out, cnt = fe.identify_fishing(df=df,  # DataFrame with the data
                                      datetime_column='Fecha_Posicion',
                                      name_column='Mmsi',  # Column with unique vessel identifier.
                                      speed_column='Sog',  # Column with vessel speed.
                                      min_trawl_speed=1.02,
                                      max_trawl_speed=4.08,
                                      min_nav_speed=4.86,
                                      max_duration_false_positive=15,
                                      max_duration_false_negative=15,
                                      min_haul=60,  # Minimum duration (mins) of a fishing haul
                                      turn_off_time=30,  # Maximum time (mins) where a gap of data is ignored
                                      remove_no_hauls=False,
                                      date_format="%d-%m-%y %H:%M"
                                      )
    assert cnt > 0 and cnt == 56
    assert all(col in df_out.columns for col in ['Haul id', 'Trawling'])
    assert ptypes.is_bool_dtype(df_out['Trawling'])
    assert ptypes.is_numeric_dtype(df_out['Haul id'])
    assert len(df) == len(df_out)

    # Check if plotting returns a figure
    fe_show.fishing_identification_check(df=df_out, name_column='Mmsi', datetime_column='Fecha_Posicion',
                                         fig_name='parameter_evaluation', dir_out='tests',
                                         format_fig='jpg', n_fig=9,
                                         latitude='Latitud_decimal', longitude='Longitud_decimal',
                                         date_format="%d-%m-%y %H:%M")
    assert os.path.exists(os.path.join('tests', 'parameter_evaluation.jpg'))
