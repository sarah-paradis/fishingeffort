import fishingeffort.fishingeffort as fe
import os
import geopandas as gpd


def test_point_to_line():
    df = fe.read_df(file='AIS_data_test_hauls_identified.csv', file_dir='tests')

    utm_zone = fe.utm_zone_epsg(df=df, latitude='Latitud_decimal', longitude='Longitud_decimal')
    assert isinstance(utm_zone, int)

    df_hauls = fe.point_to_line(df=df, name_column='Mmsi',
                                latitude='Latitud_decimal', longitude='Longitud_decimal',
                                output_crs='epsg: 32631')
    assert isinstance(df_hauls, gpd.GeoDataFrame)
    assert df_hauls.crs.to_epsg() == utm_zone
    assert df_hauls.geom_type[0] == 'LineString'
    assert df_hauls.shape[0] == df['Haul id'].nunique()


