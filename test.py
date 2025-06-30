import fishingeffort.fishingeffort as fe
import fishingeffort.show as fe_show
import pandas as pd
import numpy as np
import geopandas as gpd
import os

# Workflow
# 1. Read data
# 2. Establish trawling speed ranges
# 3. Identifying trawling activity based on speed
# 4. Convert point to line based on Haul id
# 5. Calculate line density given a specified grid size

# Global variables
min_haul = 60

file_dir = r'C:\Users\Sarah\Documents\2015-2020_PhD\Data\Datos Pesca\VMS_Llanca_Roses\Python_project\data'
# df = fe.read_df(file_dir=file_dir,
#                 file='vmsint_otb_2005-2007_64600_64710_64700.csv')
# df_2 = fe.read_df(file_dir=file_dir,
#                   file='vmsint_otb_2008-2018_64600_64710_64700.csv')
#
# df = df.append(df_2)
#
# df['Speed'].describe()
# df = fe.filter_speeds(df=df, speed_column='Speed', min_speed=0, max_speed=20)
#
# # df_test = fe.define_fishing_speed(df=df, speed_column='Speed', mean_trawl=2, mean_nav=12)
#
# df, cnt = fe.identify_trawling(df=df, datetime_column='Date', name_column='VesselId', speed_column='Speed',
#                                min_trawl_speed=2, max_trawl_speed=3.87, min_nav_speed=8,
#                                min_duration_false_negative=5, min_haul=min_haul, min_duration_false_positive=5,
#                                AIS_turn_off=30, remove_no_hauls=False)
# df_nohauls = df.dropna(subset=['Haul id'])
# # fe.save_all_data_months(df=df_nohauls, datetime_column='Date',
# #                         dir_output=os.path.join(file_dir, f'Output_{min_haul}'),
# #                         latitude='Latitude', longitude='Longitude', input_crs='epsg:4326')
# df['Year'] = pd.to_datetime(df['Date']).dt.year
# for year in df['Year'].unique():
#     df_year = df[df['Year'] == year]
#     print(f'Saving file {year}')
#     df_year.to_csv(os.path.join(file_dir, f'Output_{min_haul}', f'{year}.csv'))

# Read file and calculate point to line for each file
years = ['2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016',
         '2017', '2018']
for year in years:
    print(f'Opening file {year}')
    df_year = pd.read_csv(os.path.join(file_dir, f'Output_{min_haul}', f'{year}.csv'))
    gdf_year = gpd.GeoDataFrame(df_year, geometry=gpd.points_from_xy(x = df_year['Longitude'],
                                                                     y = df_year['Latitude']),
                                crs='epsg:4326')
    gdf_year.to_crs('epsg: 32631', inplace=True)
    gdf_year.to_file(os.path.join(file_dir, f'Output_{min_haul}', 'UTM_points', f'{year}.shp'))
    df_year.dropna(subset=['Haul id'], inplace=True)
    df_hauls_year = fe.point_to_line(df=df_year, name_column='VesselId',
                                     latitude='Latitude', longitude='Longitude', output_crs='epsg: 32631')
    print(f'Saving file {year}')
    df_hauls_year.to_file(os.path.join(file_dir, f'Output_{min_haul}', 'UTM_hauls', f'hauls_{year}.shp'))
    # Calculate line density
    fe.swept_area_ratio(grid_size=100, gdf=df_hauls_year, file_name=f'year_{min_haul}.tif',
                        bounds=(500000, 4600000, 650000, 4800000), gear_width=100,
                        dir_out=os.path.join(file_dir, f'Output_{min_haul}', 'fishing_effort'))

