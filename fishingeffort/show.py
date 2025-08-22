# Functions to plot the data
import geopandas as gpd
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from fishingeffort.utils import _print_with_time, _cartopy_UTM_projection
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm
import math


# Plot gaussian distribution of vessel speed
def draw_histogram(data, df_speed_params=None, speed_unit='kn', fig_name=None, format_fig='eps',
                   dir_out=None):
    """
    Creates a histogram of a dataset. Ideally used to plot speed of fishing vessels to identify trawling speed

    data = series where the data is saved (usually saved under a column named "Sog")

    df_speed_params = DataFrame where the information on the speed a trawler fishes at or navigates is given.
    If it is not specified, it will not annotate the histogram.

    fig_name = name of the graph to be saved as. If it is not specified, the histogram will not be saved.

    format_fig = format of the figure to be saved. Default is set to "eps". Also accepts "jpeg", "tiff", etc.

    """
    assert isinstance(data, pd.Series), 'data needs to be a Series'
    with _print_with_time('Plotting histogram of dataset'):
        # Create a histogram of the whole dataset
        fig, ax = plt.subplots()
        # Set style
        sns.set_style('whitegrid')
        sns.set_style('ticks')
        sns.set_context('notebook', font_scale=1.25)
        # Plot the figure
        sns.histplot(data, kde=False, color='grey', ax=ax)
        ax.set(xlabel=f'Vessel speed ({speed_unit})')
        ax.set(xlim=(0, 16))
        ax.set(ylabel='Frequency')
        y_max = max(list(h.get_height() for h in sns.histplot(data).patches))
        ax.set(ylim=(0, y_max * 1.5))
        if df_speed_params is not None:
            color = 'firebrick'
            # Establish parameters of trawling
            avg_speed_trawling = df_speed_params['mean']['trawling']
            min_speed_trawling = df_speed_params['range_lower']['trawling']
            max_speed_trawling = df_speed_params['range_upper']['trawling']
            std_dev_trawling = df_speed_params['std_dev']['trawling']

            # Annotate the graph with trawling information
            ax.annotate('Trawling', (avg_speed_trawling, y_max * 1.25), color=color, size=13.5,
                        va='center', ha='center')
            ax.annotate(f'{min_speed_trawling:.1f} - {max_speed_trawling:.1f} {speed_unit}',
                        (avg_speed_trawling, y_max * 1.15), color=color, size=10,
                        va='center', ha='center')

            # Plot Gaussian distribution of trawling
            def gauss(x, mu, sigma, A):
                return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

            x_plot = np.linspace(0, max_speed_trawling * 1.5, 500)
            y_mode = gauss(x_plot, avg_speed_trawling, std_dev_trawling, y_max)  # scale for visualization
            ax.plot(x_plot, y_mode, color=color, lw=2)

            # Plot limits where we consider trawling speeds
            ax.axvline(min_speed_trawling, color=color, linestyle='--')
            ax.axvline(max_speed_trawling, color=color, linestyle='--')

            # Establish parameters of navigating
            avg_speed_nav = df_speed_params['mean']['navigating']
            min_speed_nav = df_speed_params['range_lower']['navigating']
            max_speed_nav = df_speed_params['range_upper']['navigating']

            # Annotate the graph with navigating information
            ax.annotate('Navigating', (avg_speed_nav, y_max * 1.25), color='k', size=13.5,
                        va='center', ha='center')
            ax.annotate(f'{min_speed_nav:.1f} - {max_speed_nav:.1f} {speed_unit}',
                        (avg_speed_nav, y_max * 1.15), color='k', size=10,
                        va='center', ha='center')

            if 'drifting' in df_speed_params.index:
                avg_speed_drift = df_speed_params['mean']['drifting']
                min_speed_drift = df_speed_params['range_lower']['drifting']
                max_speed_drift = df_speed_params['range_upper']['drifting']

                # Annotate the graph with navigating information
                ax.annotate('Drifting', (avg_speed_drift, y_max * 1.25), color='k', size=13.5,
                            va='center', ha='center')
                ax.annotate(f'{min_speed_drift:.1f} - {max_speed_drift:.1f} {speed_unit}',
                            (avg_speed_drift, y_max * 1.25), color='k', size=10,
                            va='center', ha='center')
        fig = ax.get_figure()
        fig.tight_layout()
    if fig_name is not None:
        with _print_with_time('Saving histogram as ' + fig_name + '.' + format_fig):
            file_name = fig_name + '.' + format_fig
            if dir_out is not None:
                fig.savefig(os.path.join(dir_out, file_name), format=format_fig, dpi=500)
            else:
                fig.savefig(file_name, format=format_fig, dpi=500)


def fishing_identification_check(df, name_column, datetime_column, fig_name=None,
                                 dir_out=None, format_fig='eps', n_fig=10,
                                 latitude=None, longitude=None, input_crs='epsg:4326',
                                 date_format=None):
    import random
    assert n_fig < 25, 'Maximum number of subplots to evaluate is 25'
    df[datetime_column] = pd.to_datetime(df[datetime_column], format=date_format)
    df['Date_day_temp'] = df[datetime_column].dt.date
    if not isinstance(df, gpd.GeoDataFrame):
        with _print_with_time('Converting DataFrame into a GeoDataFrame'):
            assert all([longitude, latitude, input_crs]) is not None, \
                f'No data provided for longitude, latitude, or input_crs'
            gdf = gpd.GeoDataFrame(df,
                                   geometry=gpd.points_from_xy(df[longitude], df[latitude]),
                                   crs=input_crs)
    else:
        gdf = df
    with _print_with_time('Creating figures'):
        rows, cols = _compute_subplot_grid_size(n_fig)

        utm_proj = _cartopy_UTM_projection(gdf)

        fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 3, rows * 3),
                                constrained_layout=True,
                                subplot_kw=dict(projection=utm_proj))
        # Flatten axes to always return a 1D list-like
        axs = axs.flatten() if isinstance(axs, (list, np.ndarray)) else [axs]

        gdf.to_crs(utm_proj.proj4_init, inplace=True)

        # Define extent of the maps
        minx, miny, maxx, maxy = gdf.total_bounds
        # buffer_x = (maxx - minx) * 0.1  # 10% of the width
        # buffer_y = (maxy - miny) * 0.1  # 10% of the height
        # minx -= buffer_x
        # maxx += buffer_x
        # miny -= buffer_y
        # maxy += buffer_y

        land = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                            edgecolor='face', alpha=0.5,
                                            facecolor='lightgreen')

        vessel_day_combination = []
        for ax in axs:
            # Add general parameters
            ax.set_extent([minx, maxx, miny, maxy], crs=utm_proj)
            ax.add_feature(land)
            ax.coastlines(resolution='10m', linewidth=0.5)
            # Randomly choose n_fig different vessels and days to plot the data
            vessel = random.choice(gdf[name_column].unique())
            day = random.choice(gdf['Date_day_temp'].unique())
            gdf_filter = gdf[(gdf[name_column] == vessel) & (gdf['Date_day_temp'] == day)]
            while (vessel, day) in vessel_day_combination or gdf_filter.empty or \
                    not (True in gdf_filter['Trawling'].unique()) and (False in gdf_filter['Trawling'].unique()):
                vessel = random.choice(gdf[name_column].unique())
                day = random.choice(gdf['Date_day_temp'].unique())
                gdf_filter = gdf[(gdf[name_column] == vessel) & (gdf['Date_day_temp'] == day)]
            gdf_filter.plot(column='Trawling', categorical=True, legend=True, ax=ax, zorder=5,
                            # edgecolor='grey', linewidth=0.075,
                            legend_kwds={"title": "Trawling"})
            vessel_day_combination.append((vessel, day))
            ax.set_title(f'Vessel {str(vessel)}\n({day.strftime("%d %B %Y")})', fontsize=14)

    if fig_name is not None:
        with _print_with_time('Saving figure as ' + fig_name + '.' + format_fig):
            file_name = fig_name + '.' + format_fig
            if dir_out is not None:
                fig.savefig(os.path.join(dir_out, file_name), format=format_fig, dpi=500)
            else:
                fig.savefig(file_name, format=format_fig, dpi=500)


def _compute_subplot_grid_size(n):
    """
    Computes the optimal number of rows and columns for n subplots.
    Prioritizes a layout that's as square as possible, with rows ≥ columns.
    """
    # Start from the square root and adjust until it divides cleanly
    best_rows = math.floor(math.sqrt(n))
    best_cols = math.ceil(n / best_rows)

    while best_rows * best_cols < n:
        best_rows += 1
        best_cols = math.ceil(n / best_rows)

    return best_rows, best_cols
