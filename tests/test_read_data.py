import pandas as pd
import fishingeffort.fishingeffort as fe
import pandas.api.types as ptypes
import geopandas as gpd


# Testing functions that read data
def test_read_csv():
    df = fe.read_df(file='AIS_data_test.csv', file_dir='tests')
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (18542, 5)
    assert ptypes.is_object_dtype(df['Fecha_Posicion'])
    assert all(ptypes.is_numeric_dtype(df[num_col]) for num_col in ['Mmsi', 'Latitud_decimal',
                                                                    'Longitud_decimal', 'Sog'])


def test_read_xlsx():
    df = fe.read_df(file='AIS_data_test.xlsx', file_dir='tests')
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (18542, 5)
    assert ptypes.is_object_dtype(df['Fecha_Posicion'])
    assert all(ptypes.is_numeric_dtype(df[num_col]) for num_col in ['Mmsi', 'Latitud_decimal',
                                                                    'Longitud_decimal', 'Sog'])


def test_read_xls():
    df = fe.read_df(file='AIS_data_test.xls', file_dir='tests')
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (18542, 5)
    assert ptypes.is_object_dtype(df['Fecha_Posicion'])
    assert all(ptypes.is_numeric_dtype(df[num_col]) for num_col in ['Mmsi', 'Latitud_decimal',
                                                                    'Longitud_decimal', 'Sog'])


def test_read_shp():
    gdf = fe.read_df(file='AIS_data_test.shp', file_dir='tests')
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.shape == (18542, 6)
    assert ptypes.is_object_dtype(gdf['Fecha_Posi'])
    assert all(ptypes.is_numeric_dtype(gdf[num_col]) for num_col in ['Mmsi', 'Latitud_de', 'Longitud_d', 'Sog'])
    assert gdf.crs


def test_reduce_min():
    df = fe.read_df(file='AIS_data_test.csv', file_dir='tests')
    df_min = fe.data_reduction_min(df=df, datetime_column='Fecha_Posicion',
                                   name_column='Mmsi',
                                   additional_columns=['Latitud_decimal', 'Longitud_decimal', 'Sog'])
    assert all(col in df_min.columns for col in ['Fecha_Posicion_min', 'Date', 'Month', 'Time'])
    assert ptypes.is_datetime64_dtype(df['Fecha_Posicion_min'])
    # Truncate to minute precision
    df_min['minute'] = df_min['Fecha_Posicion_min'].dt.floor('min')
    duplicates = df_min.duplicated(subset=['Mmsi', 'minute'], keep=False)
    duplicates_df = df_min[duplicates]
    assert duplicates_df.empty, 'Output of data_reduction_min() has duplicate entries per minute'

