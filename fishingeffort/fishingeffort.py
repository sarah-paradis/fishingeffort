# Functions to calculate fishing effort

import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import geopandas as gpd
from shapely.geometry import Polygon, box, LineString
import math
from geocube.api.core import make_geocube

from fishingeffort.utils import _print_with_time, _UTM_zone_epsg
import warnings

pd.options.mode.chained_assignment = None  # default='warn'


def read_all_data(file_dir, file_type='.csv', sep=None):
    """
    Function that reads all the files (Excels) in a directory as one DataFrame.

    file_dir = file directory (string)
    """
    if not file_type.startswith('.'):
        file_type = '.' + file_type
    assert file_type in ['.csv', '.xlsx', '.xls', '.shp']
    df = pd.DataFrame()  # Create empty DataFrame
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        for file in os.listdir(file_dir):  # Open all files in the directory
            if file.endswith(file_type):
                data = read_df(file=file, file_dir=file_dir, sep=sep)
                if df.empty:
                    df = data
                else:
                    df = pd.concat([df, data], ignore_index=True)
        df = df.reset_index()
    return df


def save_df(df, file, dir_output=None):
    """
    Saves DataFrame (.xls or .csv)
    """
    # Makes sure the output directory (file_dir) exists, if not, it creates it.
    if dir_output is not None:
        os.makedirs(dir_output, exist_ok=True)
        file = os.path.join(dir_output, file)

    file_type = file[-4:]
    if file_type in ['.csv', '.xls']:
        with _print_with_time('Saving file as ' + file):
            if file_type == '.xls':
                df.to_excel(file, index=None)
            if file_type == '.csv':
                df.to_csv(file, index=None)
    else:
        raise TypeError('Data not .xls or .csv type')


def read_df(file, sep=None, file_dir=None):
    """
    Reads a file (either csv or excel) as a DataFrame

    """
    if file_dir is not None:
        file = os.path.join(file_dir, file)
    file_type = file[-4:]
    if file_type in ['.csv', '.xls', 'xlsx', '.shp']:
        with _print_with_time('Opening file ' + file):
            if file_type == '.csv':
                df = pd.read_csv(file, sep=sep, engine='python')
            elif file_type == '.xls':
                df = pd.read_excel(file, engine="xlrd")
            elif file_type == 'xlsx':
                df = pd.read_excel(file, engine="openpyxl")
            elif file_type == '.shp':
                df = gpd.read_file(file)
                assert df.crs, 'Shapefile is missing CRS'
    else:
        raise TypeError('File not recognized as a .csv, .xlsx or .xls')
    return df


def save_all_data_months(df, datetime_column, dir_output='Output', output_type='csv',
                         latitude=None, longitude=None, input_crs='epsg:4326'):
    """
    Exports a DataFrame into Excel (.xls) files separated by month, such as "2019_02.xls".
    If the number of rows (entries) exceeds the maximum Excel file size, the file is separated into several Excel files
    and the name of the file indicates the maximum day included in that file, such as "2019_02_16.xls"

    df = DataFrame to be exported

    dir_output = directory where the data will be saved. If None, saves in the same working directory. If the
    directory doesn't exist, it creates it.

    output_type = type of output file, either Excel (.xls or .xlsx) or csv, or shapefile. Default is csv.
    """

    # Makes sure the output directory (file_dir) exists, if not, it creates it.
    os.makedirs(dir_output, exist_ok=True)

    # Extract the months in the files to later create individual dataframes
    # and the names of the output .csv files for each month

    if 'Month' not in df.columns:
        df['Month'] = df[datetime_column].dt.to_period('M')

    if not isinstance(df['Month'].dtype, str):
        months = sorted(list(df.Month.unique()))  # list of the months from which the DataFrames will be extracted
        months_name = [month.strftime('%Y_%m') for month in
                       months]  # list of the names of the months for the .csv files
    else:
        months = sorted(list(df.Month.unique()))
        months_name = months
    days = []
    for month in months:
        day_month = sorted(list(df[datetime_column][df['Month'] == month].unique()))
        days.append(day_month)

    # Exporting files
    if not output_type.startswith("."):
        output_type = "." + output_type

    if output_type == '.csv':
        max_rows = np.inf  # no maximum rows in a csv file
    elif output_type == '.shp':
        max_rows = np.inf
        if not isinstance(df, gpd.geodataframe.GeoDataFrame):
            assert all([longitude, latitude, input_crs]) is not None, \
                f'No data provided for longitude, latitude, or input_crs'
            df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[longitude], df[latitude]), crs=input_crs)
    elif output_type == '.xlsx':
        max_rows = 1000000  # actually it is 1048576
    elif output_type == '.xls':
        max_rows = 65000  # actually it is 65536 (2^16)
    else:
        raise TypeError(output_type + " is not a valid output type")

    for month, month_name, idx in zip(months, months_name, range(len(months))):
        df_month = df[df['Month'] == month]  # Create a new DataFrame with the data of that specific month
        file_name = month_name + output_type
        if len(df_month) < max_rows:
            with _print_with_time('Exporting ' + file_name + ' to file'):
                if output_type == '.xls' or output_type == '.xlsx':
                    df_month.to_excel(os.path.join(dir_output, file_name), index=None)
                elif output_type == '.csv':
                    df_month.to_csv(os.path.join(dir_output, file_name), index=None)
                elif output_type == '.shp':
                    df_month.to_file(os.path.join(dir_output, file_name), index=None)
        else:
            assert output_type != '.csv'
            with _print_with_time('Exporting %s in different %s files' % (file_name, output_type)):
                while len(days[idx]) > 0:  # save files
                    for i in range(len(days[idx])):  # days
                        if len(df_month[df_month['Date'].isin(days[idx][:i + 1])]) > max_rows:
                            i = i - 1
                            break
                    if type(df['Month'][0]) is not str:
                        file_name = month_name + '_' + days[idx][i].strftime('%d') + output_type
                    else:
                        file_name = month_name + '_' + days[idx][i][-2:] + output_type
                    print('\n Exporting ' + file_name)
                    file_name = os.path.join(dir_output, file_name)
                    df_save = df_month[df_month['Date'].isin(days[idx][:i + 1])]
                    assert len(df_save) > 0, "Too many rows for day %s. Cannot save in %s" % (days[idx][
                                                                                                  i + 1].strftime('%d'),
                                                                                              output_type)
                    df_save.to_excel(file_name, index=None)
                    days[idx] = days[idx][i + 1:]


def save_all_data_years(df, datetime_column, dir_output='Output', output_type='csv',
                        latitude=None, longitude=None, input_crs=None):
    """
    Exports a DataFrame into Excel (.xls) files separated by year, such as "2019.xls".
    If the number of rows (entries) exceeds the maximum Excel file size, the file is separated into several Excel files
    and the name of the file indicates the maximum day included in that file, such as "2019_02_16.xls"

    df = DataFrame to be exported

    dir_output = directory where the data will be saved. If None, saves in the same working directory. If the
    directory doesn't exist, it creates it.

    output_type = type of output file, either Excel (.xls or .xlsx) or csv, or shapefile. Default is csv.
    """

    # Makes sure the output directory (file_dir) exists, if not, it creates it.
    os.makedirs(dir_output, exist_ok=True)

    # Extract the years in the files to later create individual dataframes
    # and the names of the output .csv files for each year

    if 'Year' not in df.columns:
        df['Year'] = df[datetime_column].dt.to_period('Y')

    if not isinstance(df['Year'].dtype, str):
        years = sorted(list(df.Year.unique()))  # list of the years from which the DataFrames will be extracted
        years_name = [year.strftime('%Y') for year in years]  # list of the names of the years for the .csv files
    else:
        years = sorted(list(df.Year.unique()))
        years_name = years
    days = []
    for year in years:
        day_year = sorted(list(df[datetime_column][df['Year'] == year].unique()))
        days.append(day_year)

    # Exporting files
    if not output_type.startswith("."):
        output_type = "." + output_type

    if output_type == '.csv':
        max_rows = np.inf  # no maximum rows in a csv file
    elif output_type == '.shp':
        max_rows = np.inf
        if not isinstance(df, gpd.geodataframe.GeoDataFrame):
            assert all([longitude, latitude, input_crs]) is not None, \
                f'No data provided for longitude, latitude, or input_crs'
            df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[longitude], df[latitude]), crs=input_crs)
    elif output_type == '.xlsx':
        max_rows = 1000000  # actually it is 1048576
    elif output_type == '.xls':
        max_rows = 65000  # actually it is 65536 (2^16)
    else:
        raise TypeError(output_type + " is not a valid output type")

    for year, year_name, idx in zip(years, years_name, range(len(years))):
        df_year = df[df['Year'] == year]  # Create a new DataFrame with the data of that specific year
        file_name = year_name + output_type
        if len(df_year) < max_rows:
            with _print_with_time('Exporting ' + file_name + ' to file'):
                if output_type == '.xls' or output_type == '.xlsx':
                    df_year.to_excel(os.path.join(dir_output, file_name), index=None)
                elif output_type == '.csv':
                    df_year.to_csv(os.path.join(dir_output, file_name), index=None)
                elif output_type == '.shp':
                    df_year.to_file(os.path.join(dir_output, file_name), index=None)
        else:
            assert output_type != '.csv'
            with _print_with_time('Exporting %s in different %s files' % (file_name, output_type)):
                while len(days[idx]) > 0:  # save files
                    for i in range(len(days[idx])):  # days
                        if len(df_year[df_year['Date'].isin(days[idx][:i + 1])]) > max_rows:
                            i = i - 1
                            break
                    if type(df['Year'][0]) is not str:
                        file_name = year_name + '_' + days[idx][i].strftime('%d') + output_type
                    else:
                        file_name = year_name + '_' + days[idx][i][-2:] + output_type
                    print('\n Exporting ' + file_name)
                    file_name = os.path.join(dir_output, file_name)
                    df_save = df_year[df_year['Date'].isin(days[idx][:i + 1])]
                    assert len(df_save) > 0, "Too many rows for day %s. Cannot save in %s" % (days[idx][
                                                                                                  i + 1].strftime('%d'),
                                                                                              output_type)
                    df_save.to_excel(file_name, index=None)
                    days[idx] = days[idx][i + 1:]


def save_all_data(df, file_name, dir_output='Output', output_type='csv',
                  latitude=None, longitude=None, input_crs=None):
    """
    Exports a DataFrame into Excel (.xls) files.

    df = DataFrame to be exported

    dir_output = directory where the data will be saved. If None, saves in the same working directory. If the
    directory doesn't exist, it creates it.

    output_type = type of output file, either Excel (.xls or .xlsx) or csv, or shapefile. Default is csv.
    """

    # Makes sure the output directory (file_dir) exists, if not, it creates it.
    os.makedirs(dir_output, exist_ok=True)

    # Exporting files
    if not output_type.startswith("."):
        output_type = "." + output_type

    if output_type == '.csv':
        max_rows = np.inf  # no maximum rows in a csv file
    elif output_type == '.shp':
        max_rows = np.inf
        if not isinstance(df, gpd.geodataframe.GeoDataFrame):
            assert all([longitude, latitude, input_crs]) is not None, \
                f'No data provided for longitude, latitude, or input_crs'
            df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[longitude], df[latitude]), crs=input_crs)
            date_columns = [column for column, type in df.dtypes.iteritems() if type == '<M8[ns]']
            for date_column in date_columns:
                df[date_column] = df[date_column].astype(str)
    elif output_type == '.xlsx':
        max_rows = 1000000  # actually it is 1048576
    elif output_type == '.xls':
        max_rows = 65000  # actually it is 65536 (2^16)
    else:
        raise TypeError(output_type + " is not a valid output type")

    if len(df) < max_rows:
        with _print_with_time('Exporting ' + file_name + ' to file'):
            if output_type == '.xls' or output_type == '.xlsx':
                df.to_excel(os.path.join(dir_output, file_name), index=None)
            elif output_type == '.csv':
                df.to_csv(os.path.join(dir_output, file_name), index=None)
            elif output_type == '.shp':
                df.to_file(os.path.join(dir_output, file_name), index=None)
    else:
        assert output_type != '.csv'
        print('Can not save output file')


def data_reduction_min(df, datetime_column, name_column,
                       additional_columns, date_format=None):
    """
    Extracts the first entry of each minute for
    each vessel during the sampling period.

    df = DataFrame that needs to be processed

    datetime_column = name of column in dataframe that has the date and
    time of vessel positioning (type: string)

    name_column = name of column in dataframe that has the name/code of
    each vessel

    additional_columns = list with the name of the columns in the dataframe
    that would be exported

    Returns a dataframe with data every minute
    """

    # Create new date-time column rounded to minutes
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_column]):
        with _print_with_time(f'Converting column {datetime_column} to Datetime'):
            # Convert datetime_column to datetime
            df[datetime_column] = pd.to_datetime(df[datetime_column], format=date_format)
    new_column = datetime_column + '_min'
    df[new_column] = df[datetime_column]
    with _print_with_time('Rounding time to minutes'):
        df[new_column] = df[new_column].values.astype('<M8[m]')

    # Extract the first entry of each minute for each vessel and each day
    with _print_with_time('Extracting the first entry of each minute for each vessel and day'):
        df_min = df.groupby([name_column, new_column]).first()

    df_min = df_min.reset_index()  # Reset index

    # Organize DataFrame eliminating the unnecessary columns and putting them in order
    df_min = df_min[[name_column] + [datetime_column] + [new_column] + additional_columns]

    # Create a new column with only the month and year
    with _print_with_time('Grouping data by month and day'):
        df_min['Month'] = df_min[new_column].dt.to_period('M')
        df_min['Date'] = df_min[new_column].dt.date
        df_min['Time'] = df_min[new_column].dt.time

        # Extract the months in the files to later create individual DataFrames
        # and the names of the output .csv files for each month
        months = sorted(list(df_min.Month.unique()))  # list of the months from which the DataFrames will be extracted
        months_name = [month.strftime('%Y_%m') for month in
                       months]  # list of the names of the months for the .csv files
        days = []
        for month in months:
            day_month = sorted(list(df_min['Date'][df_min['Month'] == month].unique()))
            days.append(day_month)
    return df_min


def save_all_vessels(df, name_column, output_name, file_dir=None):
    """
    Creates an Excel file of all the vessels that were found in the DataSet

    df = name of complete DataFrame. If the data is separated into different files, open them all using
    the read_all_data function found in read_all_data.py

    name_column = column name where the name of the vessels are stored

    output_name = name of Excel file with all the vessels in the DataSet

    file_dir = directory where the Excel will be saved as
    """
    if output_name.endswith('.csv'):
        output_name = output_name.split('.csv')[0]
    if not output_name.endswith(".xlsx"):
        output_name += ".xlsx"
    if file_dir is not None:
        # makes sure the output directory (file_dir) exists, if not, it creates it.
        os.makedirs(file_dir, exist_ok=True)
        # joins path name of the directory with the output file name if the output directory file is given
        file_name = os.path.join(file_dir, output_name)
    else:
        file_name = output_name
    with _print_with_time('Extracting vessels in the area'):
        # Extract the names of the vessels
        # Extract the first entry of all vessels, to get their identifier
        df_vessels = df.groupby([name_column]).first()
        df_vessels = df_vessels.reset_index()  # Reset index
        df_vessels.to_excel(file_name, index=None)
    return df_vessels


def additional_data(df, name_column, file_fleet, Fleet_column):
    """
    Add all the additional data (Gear type, Gt, Power (kw), Construction year,
    Port). Additional data are extracted from Fleet Register on the Net:
        http://ec.europa.eu/fisheries/fleet/index.cfm

    df = Dataframe with data from all fishing vessels

    name_column = name of column in original DataFrame that has the name/code of
    each vessel.

    file_fleet = file from Fleet Register that has all the data to be extracted

    Fleet_column = name of column in Fleet Register on the Net that has the
    name/code of each vessel.

    Returns a DataFrame
    """

    # Fishing fleet on the net
    global df_vessel_info
    if file_fleet.endswith('csv'):
        df_vessel_info = pd.read_csv(file_fleet, engine='python', delimiter=';')
    elif file_fleet[-3:] == 'xls' or file_fleet[-4:] == 'xlsx':
        df_vessel_info = pd.read_excel(file_fleet)

    # ADDING THE ADDITIONAL DATA IN THE ORIGINAL DATAFRAME
    with _print_with_time('Adding additional data to the DataFrame'):
        # Create dictionary of the data we want to add from Fishing Fleet on the Net
        gear_main = df_vessel_info.groupby([Fleet_column])['Gear Main Code'].last().to_dict()
        gear_sec = df_vessel_info.groupby([Fleet_column])['Gear Sec Code'].last().to_dict()
        ton_gt = df_vessel_info.groupby([Fleet_column])['Ton Gt'].last().to_dict()
        ton_oth = df_vessel_info.groupby([Fleet_column])['Ton Oth'].last().to_dict()
        power_main = df_vessel_info.groupby([Fleet_column])['Power Main'].last().to_dict()
        power_aux = df_vessel_info.groupby([Fleet_column])['Power Aux'].last().to_dict()
        const_year = df_vessel_info.groupby([Fleet_column])['Construction Year'].last().to_dict()
        port = df_vessel_info.groupby([Fleet_column])['Port Name'].last().to_dict()

        # Create new column based on data from Fishing Fleet on the Net
        df['Gear Main'] = df[name_column].map(gear_main)
        df['Gear Sec'] = df[name_column].map(gear_sec)
        df['Ton Gt'] = df[name_column].map(ton_gt)
        df['Ton other'] = df[name_column].map(ton_oth)
        df['Power Main'] = df[name_column].map(power_main)
        df['Power Aux'] = df[name_column].map(power_aux)
        df['Construction Year'] = df[name_column].map(const_year)
        df['Port Name'] = df[name_column].map(port)

    return df


def no_data_fleet(df, column_name, vessel_name, output_name, file_dir=None):
    """
    Extracts an Excel with the name of all the vessels that were not paired with the data from Fleet Register database

    df = DataFrame that has already been paired with Fleet Register database

    column_name = name of column where parameters from Fleet Register were added

    vessel_name = name of column with the Vessel name/code

    output_name = name of Excel file to be exported

    file_dir = name of directory where the Excel file will be saved in. If None, the Excel file will be saved in the
    same working directory

    Returns a DataFrame
    """

    if not output_name.endswith(".xlsx"):
        output_name += ".xlsx"
    if file_dir is not None:
        # makes sure the output directory (file_dir) exists, if not, it creates it.
        os.makedirs(file_dir, exist_ok=True)
        # joins path name of the directory with the output file name if the output directory file is given
        file_name = os.path.join(file_dir, output_name)
    else:
        file_name = output_name

    with _print_with_time('Extracting vessels with no data in Fleet Register'):
        df_no_data = df[df.loc[:, column_name].isna()]
        # Extracts only the first entry of the vessels that were not included
        df_no_data = df_no_data.groupby(vessel_name).first()
        df_no_data = df_no_data.reset_index()
        df_no_data.to_excel(file_name, index=None)
    return df_no_data


def filter_trawlers(df, column_gear, gear_name='OTB'):
    """
    Filters DataFrame for a specific gear type. Default extracts vessels with "OTB" (otter trawl boards).

    df = DataFrame with the whole dataset

    column_gear = name of column that has the fishing gear type.
    If there are multiple columns, introduce names as a list

    gear_name = name of the gear type to be extracted. Default is 'OTB'.
    """
    with _print_with_time('Extracting gear type'):
        # Extract by gear type
        assert type(column_gear) in (list, str), "Data with fishing gear is not correctly given"
        if type(column_gear) is list:
            # Checks if any of the columns has the gear_name as fishing gear
            df_trawl = df[(df.loc[:, column_gear] == gear_name).any(axis=1)]
        else:
            # Checks if vessel fishing gear has the gear_name
            df_trawl = df[df.loc[:, column_gear] == gear_name]
        # Reset index since filtering out samples must have altered the index of DataFrame
        df_trawl = df_trawl.reset_index().drop('index', axis=1)
    return df_trawl


def define_fishing_speed(df, speed_column,
                         mean_drift=None, mean_trawl=3, mean_nav=10,
                         mode='bimodal'):
    """
    Fits Gaussian distributions to vessel speed data with optional bimodal or trimodal modes.

    Parameters:
    - df: DataFrame with speed data
    - speed_column: column name with speeds
    - mean_drift: initial guess for drifting speed (only for trimodal)
    - mean_trawl: initial guess for trawling speed
    - mean_nav: initial guess for navigating speed
    - mode: 'unimodal', 'bimodal' or 'trimodal'. Default is 'bimodal'

    Returns:
    - DataFrame with parameters and 95% confidence ranges
    """
    data = df[speed_column].dropna()

    # Histogram setup
    y = np.array(data.value_counts(bins=100, sort=False))
    bin_edges = np.linspace(min(data), max(data), num=101)
    x = (bin_edges[1:] + bin_edges[:-1]) / 2

    def gauss(x, mu, sigma, A):
        return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    def unimodal(x, mu1, sigma1, A1):
        return gauss(x, mu1, sigma1, A1)

    def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
        return gauss(x, mu1, sigma1, A1) + gauss(x, mu2, sigma2, A2)

    def trimodal(x, mu1, sigma1, A1, mu2, sigma2, A2, mu3, sigma3, A3):
        return gauss(x, mu1, sigma1, A1) + gauss(x, mu2, sigma2, A2) + gauss(x, mu3, sigma3, A3)

    if mode == 'bimodal':
        expected = [mean_trawl, 1, max(y[:50]), mean_nav, 1, max(y[50:])]
        params, _ = curve_fit(bimodal, x, y, p0=expected)
        means = [params[0], params[3]]
        stds = [params[1], params[4]]
        labels = ['trawling', 'navigating']

    elif mode == 'unimodal':
        expected = [mean_trawl, 1, max(y[:50])]
        params, _ = curve_fit(unimodal, x, y, p0=expected)
        means = [params[0]]
        stds = [params[1]]
        labels = ['trawling']

    elif mode == 'trimodal':
        if mean_drift is None:
            raise ValueError("You must provide 'mean_drift' when using trimodal mode.")
        # Initial amplitudes roughly proportional to thirds of histogram
        expected = [mean_drift, 1, max(y[:33]),
                    mean_trawl, 1, max(y[33:66]),
                    mean_nav, 1, max(y[66:])]
        params, _ = curve_fit(trimodal, x, y, p0=expected)
        means = [params[0], params[3], params[6]]
        stds = [params[1], params[4], params[7]]
        labels = ['drifting', 'trawling', 'navigating']

    else:
        raise ValueError("mode must be 'bimodal' or 'trimodal'")

    df_params = pd.DataFrame({
        'mean': means,
        'std_dev': stds,
        'range_lower': [m - 1.96 * s for m, s in zip(means, stds)],
        'range_upper': [m + 1.96 * s for m, s in zip(means, stds)]
    }, index=labels)

    return df_params


def _classify_speed(df, speed_column, min_trawl, max_trawl, min_nav):
    """ Classifies Speed Over Ground (SOG) into low speed (0), medium speed (1), high speed (2), and navigating speed (3)
    where medium speed is the trawler speed when operating (doing a haul)
    sog_col = column where SOG is saved as in the DataFrame
    min_trawl = minimum trawling speed
    max_trawl = maximum trawling speed
    min_nav = minimum navigating speed
    """
    with _print_with_time('Classifying by Speed Over Ground'):
        class_sog = np.zeros(len(df[speed_column]), dtype=int)
        class_sog[(df[speed_column] >= min_trawl) & (df[speed_column] <= max_trawl)] = 1
        class_sog[(df[speed_column] > max_trawl) & (df[speed_column] < min_nav)] = 2
        class_sog[df[speed_column] >= min_nav] = 3
    return class_sog


def _get_chunk_indices(a, in_same_chunk_fn=lambda x, y: x == y):
    len_a = len(a)
    diffs = [not in_same_chunk_fn(a.iloc[i], a.iloc[i + 1]) for i in range(len_a - 1)]
    indices = np.arange(len_a)
    start_indices = indices[1:][diffs].tolist()  # index of the first different value
    start_indices.insert(0, 0)  # insert first index
    end_indices = indices[:-1][diffs].tolist()  # index of the last 'equal' value of the chunk
    end_indices.append(len_a - 1)  # insert last index
    return list(zip(start_indices, end_indices))


def _get_same_period_fn(minutes_diff, criteria, datetime_column):
    def same_period(x, y):
        """ Evaluates entries that are registered in less than 'minutes_diff' in order to decide if they
        belong in the same chunk

        minutes_diff = time interval to check a chunk

        criteria  = list of column names to check equalness

        datetime_column = column with datetime information
        """
        time_diff = pd.Timedelta(minutes=minutes_diff)
        res = y[datetime_column] - x[datetime_column] < time_diff

        for crit in criteria:
            res = res and (x[crit] == y[crit])
        return res

    return same_period


def filter_speeds(df, speed_column, min_speed=0, max_speed=20):
    # Filters out vessel speeds that are impossible
    df = df[(df[speed_column] < max_speed) & (df[speed_column] > min_speed)]
    df.reset_index(drop=True, inplace=True)
    return df


def identify_fishing(df, datetime_column, name_column, speed_column, min_trawl_speed, max_trawl_speed,
                     min_nav_speed, max_duration_false_positive, max_duration_false_negative, min_haul,
                     turn_off_time, remove_no_hauls=False, start_trawl_id=1, date_format=None):
    """
    Identifies trawling and haul ids.
    First:
    Find false-positives: when vessel is navigating at trawling speed but
    not doing a haul.
    This is corrected by establishing a minimum trawling duration (min_haul)
    Then:
    Find false-negatives: when vessel decreases/increases speed below/above
    trawling threshold but is actually still trawling.
    These are identified by establishing a maximum time that the vessel can
    be doing a haul at a slower speed (min_duration_false_negative)
    Finally:
    Creates new columns (Trawl, Haul_id) establishing whether vessel is trawling and the haul ID

    Parameters:
    df = DataFrame
    datetime_column = name of column in dataframe that has the date and time of vessel positioning (type: string)
    name_column = name of column in dataframe with the name/code ID of the vessel
    max_duration_false_positive = duration of continued entries classified as trawling need to take place
    max_duration_false_negative = duration that these events take in minutes (maximum duration)
    min_haul = minimum duration of a haul (time in minutes)
    turn_off_time = maximum time that data is missing before considering it belongs to a different haul

    Returns DataFrame with the additional columns of 'Sog criteria', 'Trawling'
    and 'Haul id'
    """

    if not pd.api.types.is_datetime64_any_dtype(df[datetime_column]):
        with _print_with_time(f'Converting column {datetime_column} to Datetime'):
            # Convert datetime_column to datetime
            df[datetime_column] = pd.to_datetime(df[datetime_column], format=date_format)
    # Sort by datetime and vessel_id before filtering
    df.sort_values(by=[name_column, datetime_column], inplace=True)

    # Classify speed based on criteria
    df['speed_criteria'] = _classify_speed(df=df, speed_column=speed_column, min_trawl=min_trawl_speed,
                                           max_trawl=max_trawl_speed, min_nav=min_nav_speed)

    # Create extra columns that Trawling and Haul_id that will be needed in the future
    df['Date_day'] = df[datetime_column].dt.date
    df['speed_criteria_column_temp'] = df['speed_criteria']
    df['Trawling'] = np.zeros(len(df), dtype=bool)
    df['Haul id'] = np.full(len(df), np.nan, dtype=np.float32)

    # Reset index to make sure that the index goes from 0 to n_rows (to assign values in .iloc)
    df.reset_index(inplace=True, drop=True)

    with _print_with_time('Getting set of trawling criteria'):
        # Creates a list of tuples (index start, index end) of all the 'chunks' based on same SOG criteria (0,1,2)
        # considering that the vessel's AIS has been turned off during less than 'AIS_turn_off'.
        classify_trawling_list = _get_chunk_indices(df, _get_same_period_fn(turn_off_time, ['speed_criteria',
                                                                                            name_column,
                                                                                            'Date_day'],
                                                                            datetime_column))
    if max_duration_false_positive > 0:
        with _print_with_time('Identifying false-positives'):
            # Converts min_haul into datetime format
            max_duration_false_positive = pd.Timedelta(minutes=max_duration_false_positive)
            for chunk in classify_trawling_list:
                start = chunk[0]
                end = chunk[1]
                if df['speed_criteria'].loc[start] == 1:
                    if (df[datetime_column].loc[end] - df[datetime_column].loc[start]) > max_duration_false_positive:
                        # The duration was longer than the maximum duration of false positive,
                        # so it is considered to actually be fishing and the speed_criteria is kept at 1 (fishing)
                        df.loc[start:end, 'speed_criteria_column_temp'] = 1
                        assert all(df['speed_criteria_column_temp'].loc[start:end] == 1)
                    else:
                        # The duration was shorter than the maximum duration of false positive,
                        # so it is considered to be a false positive. The speed_criteria is set to 0 (not fishing)
                        df.loc[start:end, 'speed_criteria_column_temp'] = 0
                        assert all(df['speed_criteria_column_temp'].loc[start:end] == 0)
    if max_duration_false_negative > 0:
        with _print_with_time('Identifying false-negatives'):
            # Convert min_duration_false_negative into minutes (time format)
            max_duration_false_negative = pd.Timedelta(minutes=max_duration_false_negative)
            # Check if 0 (low speed) or 2 (high speeds) are between 1 (fishing speed) and its duration.
            # If the duration of these reductions in speeds (between fishing events) are less
            # than the specified time criteria, it is converted into 1 (fishing speed)
            for idx in range(1, len(classify_trawling_list) - 2):
                # Checks Sog criteria of current chunk
                current_class = df['speed_criteria'].loc[classify_trawling_list[idx][1]]
                if current_class == 0 or current_class == 2:
                    # Checks Sog criteria of previous chunk
                    prev_class = df['speed_criteria_column_temp'].loc[classify_trawling_list[idx - 1][1]]
                    # Checks Sog criteria of following chunk
                    next_class = df['speed_criteria_column_temp'].loc[classify_trawling_list[idx + 1][1]]
                    if prev_class == 1 and next_class == 1:
                        # Previous and following classifications are in fishing speed
                        start, end = classify_trawling_list[idx]
                        if (df[datetime_column].loc[end] - df[datetime_column].loc[start]) \
                                <= max_duration_false_negative:
                            # The duration of the event (not at fishing speed) is shorter than the maximum duration
                            # of false-negatives, so it is considered a false-negative and the speed_criteria is
                            # changed to 1 (fishing)
                            df.loc[start:end, 'speed_criteria_column_temp'] = 1
                            assert all(df['speed_criteria_column_temp'].loc[start:end] == 1)
    if not max_duration_false_negative == 0 or not max_duration_false_positive == 0:
        with _print_with_time('Getting new set of trawling criteria'):
            # After correcting for false-positives and false-negatives, re-evaluate fishing activity.
            # Creates a list of tuples (index start, index end) of all the 'chunks' based on same SOG criteria (0,1,2)
            # considering that the vessel's GPS positioning has been turned off during less than 'turn_off_time'.
            classify_trawling_list = _get_chunk_indices(df, _get_same_period_fn(turn_off_time,
                                                                                ['speed_criteria_column_temp',
                                                                                 name_column,
                                                                                 'Date_day'],
                                                                                datetime_column))
    with _print_with_time('Identifying trawling activity after the previous corrections'):
        # Converts min_haul into datetime format
        min_haul = pd.Timedelta(minutes=min_haul)
        for chunk in classify_trawling_list:
            start = chunk[0]
            end = chunk[1]
            if df['speed_criteria_column_temp'].loc[start] == 1:
                if (df[datetime_column].loc[end] - df[datetime_column].loc[start]) > min_haul:
                    df.loc[start:end, 'Trawling'] = True
                    assert all(df['Trawling'].loc[start:end] == True)
    with _print_with_time('Identifying Haul_id'):
        cnt = 0
        new_trawling_list = _get_chunk_indices(df, _get_same_period_fn(turn_off_time,
                                                                       ['Trawling',
                                                                        name_column,
                                                                        'Date_day'],
                                                                       datetime_column))
        for chunk in new_trawling_list:
            start = chunk[0]
            end = chunk[1]
            if df['Trawling'].loc[start] == True:
                df.loc[start:end, 'Haul id'] = cnt + start_trawl_id
                assert all(df['Haul id'].loc[start:end] == cnt + start_trawl_id)
                cnt += 1
    df.drop(columns=['speed_criteria_column_temp', 'speed_criteria'], inplace=True)
    if remove_no_hauls:
        df['Haul id'].replace(0, np.nan, inplace=True)
        df.dropna(subset=['Haul id'], inplace=True)
    return df, cnt


def _create_grid(side_length, shape='square', feature=None, bounds=None, proj=None):
    """Create a grid consisting of either rectangles or hexagons with a specified side length that covers
    the extent of input feature."""
    assert feature is not None or bounds is not None, \
        'Please provide either a feature or bounds to know the extension of the grid'

    if bounds is not None:
        min_x, min_y, max_x, max_y = bounds
    elif feature is not None:
        # Get extent of buffered input feature
        min_x, min_y, max_x, max_y = feature.total_bounds
    else:
        min_x, min_y, max_x, max_y = np.nan * 4
    assert isinstance(min_x, (int, float)) and isinstance(min_y, (int, float)) \
           and isinstance(max_x, (int, float)) and isinstance(max_y, (int, float))
    assert min_x < max_x and min_y < max_y

    # Create empty list to hold individual cells that will make up the grid
    cells_list = []

    # Create grid of squares if specified
    if shape in ["square", "rectangle", "box"]:

        # Adapted from https://james-brennan.github.io/posts/fast_gridding_geopandas/
        # Create and iterate through list of x values that will define column positions with specified side length
        for x in np.arange(min_x - side_length, max_x + side_length, side_length):

            # Create and iterate through list of y values that will define row positions with specified side length
            for y in np.arange(min_y - side_length, max_y + side_length, side_length):
                # Create a box with specified side length and append to list
                cells_list.append(box(x, y, x + side_length, y + side_length))


    # Otherwise, create grid of hexagons
    elif shape == "hexagon":

        # Set horizontal displacement that will define column positions with specified side length (based on normal
        # hexagon)
        x_step = 1.5 * side_length

        # Set vertical displacement that will define row positions with specified side length (based on normal hexagon)
        # This is the distance between the centers of two hexagons stacked on top of each other (vertically)
        y_step = math.sqrt(3) * side_length

        # Get apothem (distance between center and midpoint of a side, based on normal hexagon)
        apothem = (math.sqrt(3) * side_length / 2)

        # Set column number
        column_number = 0

        # Create and iterate through list of x values that will define column positions with vertical displacement
        for x in np.arange(min_x, max_x + x_step, x_step):

            # Create and iterate through list of y values that will define column positions with horizontal displacement
            for y in np.arange(min_y, max_y + y_step, y_step):
                # Create hexagon with specified side length
                hexagon = [[x + math.cos(math.radians(angle)) * side_length, y + math.sin(math.radians(angle)) *
                            side_length] for angle in range(0, 360, 60)]

                # Append hexagon to list
                cells_list.append(Polygon(hexagon))

            # Check if column number is even
            if column_number % 2 == 0:

                # If even, expand minimum and maximum y values by apothem value to vertically displace next row
                # Expand values so as to not miss any features near the feature extent
                min_y -= apothem
                max_y += apothem

            # Else, odd
            else:

                # Revert minimum and maximum y values back to original
                min_y += apothem
                max_y -= apothem

            # Increase column number by 1
            column_number += 1

    # Else, raise error
    else:
        raise Exception("Specify a rectangle or hexagon as the grid shape.")

    # Create grid from list of cells
    if feature is not None:
        proj = feature.crs
    grid = gpd.GeoDataFrame(cells_list, columns=['geometry'], crs=proj)

    # Create a column that assigns each grid a number
    grid["Grid_ID"] = np.arange(len(grid))

    # Return grid
    return grid


def point_to_line(df, name_column, haulid_column='Haul id', date_column='Date_day',
                  input_crs='epsg:4326', latitude=None, longitude=None, output_crs=None, additional_columns=None):
    if additional_columns is not None:
        assert isinstance(additional_columns, (list, tuple)), \
            f'variable additional_columns needs to be a list or tuple of columns to be added in the output, ' \
            f'not {additional_columns}'
    with _print_with_time('Converting point to line'):
        if isinstance(df, pd.DataFrame):
            assert all([longitude, latitude, input_crs]) is not None, \
                f'No data provided for longitude, latitude, or input_crs'
            gdf = gpd.GeoDataFrame(df,
                                   geometry=gpd.points_from_xy(x=df[longitude], y=df[latitude]),
                                   crs=input_crs)
        else:
            gdf = df
        assert isinstance(gdf, gpd.GeoDataFrame) and gdf.crs == input_crs
        metadata = [name_column, haulid_column, date_column]
        if additional_columns:
            metadata.extend(additional_columns)
        gdf_hauls_lines = gdf.groupby(metadata)['geometry'].apply(lambda x: LineString(x.tolist()))
    gdf_hauls_lines = gdf_hauls_lines.reset_index()
    gdf_hauls_lines = gdf_hauls_lines.set_crs(input_crs)  # For some reason, the crs is lost
    if output_crs:
        with _print_with_time(f'Reprojecting to {output_crs}'):
            # It's actually useful to reproject to  UTM (coordinates in meters)
            return gdf_hauls_lines.to_crs(output_crs)
    else:
        return gdf_hauls_lines


def utm_zone_epsg(df, latitude, longitude, input_crs='epsg: 4326'):
    if isinstance(df, pd.DataFrame):
        assert all([longitude, latitude, input_crs]) is not None, \
            f'No data provided for longitude, latitude, or input_crs'
        gdf = gpd.GeoDataFrame(df,
                               geometry=gpd.points_from_xy(x=df[longitude], y=df[latitude]),
                               crs=input_crs)
    else:
        gdf = df
    return _UTM_zone_epsg(gdf=gdf)


def swept_area_ratio(grid_size, gdf, file_name, gear_width, crs=None, bounds=None, dir_out=None):
    assert file_name.endswith('.tif')
    with _print_with_time(f'Calculating fishing effort in {grid_size} grid size (crs units)'):
        # Create grid where line density is calculated
        grid = _create_grid(side_length=grid_size, feature=gdf, bounds=bounds, proj=crs)
        # Perform spatial join merging the information of the grid and the feature
        # Predicate = intersects also counts features that fall on a cell boundary (between two cells)
        if grid.crs != gdf.crs:
            if not grid.crs:
                grid.set_crs(gdf.crs, inplace=True)
            else:
                grid.to_crs(gdf.crs, inplace=True)
        assert grid.crs == gdf.crs
        column_name = f'Fishing_effort_{grid_size}'
        # Count the number of hauls within a specified search radius, which is half the gear width
        # (assume the vessel's position is in the middle)
        assert isinstance(gear_width, int), f'gear_width needs to be an integer'
        search_radius = int(gear_width / 2)
        hauls_buffer = gdf.buffer(distance=search_radius, cap_style=2)
        hauls_buffer.reset_index(drop=True, inplace=True)
        hauls_buffer = gpd.GeoDataFrame({'Id': np.ones(len(hauls_buffer))},
                                        geometry=hauls_buffer, crs=crs)
        gdf = hauls_buffer
        gdf_grid = gpd.sjoin(left_df=gdf, right_df=grid, how='inner', predicate='intersects')
        # Groupby and count the number of lines per grid cell.
        gdf_grid[column_name] = gear_width / grid_size  # Adjust the value based on the swept area ratio
        fishing_effort = gdf_grid.groupby('Grid_ID')[column_name].sum()
        # Merge with previous grid cell, using a left join to keep all grids
        output_grid = grid.merge(fishing_effort, on='Grid_ID', how='left')
        # Convert into a raster
        out_raster = make_geocube(vector_data=output_grid, measurements=[f"Fishing_effort_{grid_size}"],
                                  resolution=(-grid_size, grid_size))  # for most crs negative comes first in resolution
        if dir_out:
            os.makedirs(dir_out, exist_ok=True)
            out_file = os.path.join(dir_out, file_name)
        else:
            out_file = file_name
        out_raster[f"Fishing_effort_{grid_size}"].rio.to_raster(out_file)
