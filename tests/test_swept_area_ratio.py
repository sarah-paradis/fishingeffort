import os
import fishingeffort.fishingeffort as fe
import geopandas as gpd


def test_swept_area_ratio():
    gdf = fe.read_df(file_dir=os.path.join('tests', 'AIS_data_test_hauls'),
                     file='AIS_data_test_hauls.shp')
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.geom_type[0] == 'LineString'

    fe.swept_area_ratio(grid_size=100,
                        gdf=gdf,
                        file_name='SAR.tif',
                        gear_width=100,
                        dir_out='tests')
    assert os.path.exists(os.path.join('tests', 'SAR.tif'))
