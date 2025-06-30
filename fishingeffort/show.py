# Functions to plot the data
import geopandas as gpd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from fishingeffort.utils import _print_with_time

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))


# Plot gaussian distribution of vessel speed
def draw_histogram(data, speed_kwds=None, fig_name=None, format_fig='eps'):
    """
    Creates a histogram of a dataset. Ideally used to plot speed of fishing vessels to identify trawling speed

    data = series where the data is saved (usually saved under a column named "Sog")

    df_speed = DataFrame where the information on the speed a trawler operates at or navigates is given.
    If it is not specified, it will not annotate the histogram.

    fig_name = name of the graph to be saved as. If it is not specified, the histogram will not be saved.

    format_fig = format of the figure to be saved. Default is set to "eps". Also accepts "jpeg", "tiff", etc.

    """
    with _print_with_time('Plotting histogram of dataset'):
        ### Create a histogram of the whole dataset
        # Set style
        sns.set_style('whitegrid')
        sns.set_style('ticks')
        sns.set_context('notebook', font_scale=1.25)
        # Plot the figure
        ax = sns.histplot(data, kde=False, color='grey')
        ax.set(xlabel='Speed over ground')
        ax.set(xlim=(0, 16))
        ax.set(ylabel='Frequency')
        y_max = max(list(h.get_height() for h in sns.histplot(data).patches))
        ax.set(ylim=(0, y_max * 1.5))
        if speed_kwds is not None:
            if 'avg_trawl_speed' in speed_kwds.keys():
                avg_speed_trawling = speed_kwds['avg_speed_trawling']
            if 'min_trawl_speed' in speed_kwds.keys():
                min_speed_trawling = speed_kwds['min_speed_trawling']
            # TODO finish figure
            # # Establish parameters of trawling
            # avg_speed_trawling = speed_kwds['']
            # std_dev_trawling = df_speed.iloc[1, 0]
            # max_freq_trawling = df_speed.iloc[2, 0]
            # min_speed = round((df_speed.iloc[0, 0] - stds * df_speed.iloc[1, 0]), 1)
            # max_speed = round((df_speed.iloc[0, 0] + stds * df_speed.iloc[1, 0]), 1)
            #
            # # Annotate the graph with trawling information
            # ax.text((min_speed + std_dev_trawling * 0.5), y_max * 1.25, 'Trawling', color='blue', size=13.5)
            # ax.text((min_speed + std_dev_trawling * 0.5), y_max * 1.1, '%.1f - %.1f kn' % (min_speed, max_speed),
            #         color='blue', size=10)
            #
            # # Establish parameters of navigating
            # avg_speed_nav = df_speed.iloc[3, 0]
            # std_dev_nav = df_speed.iloc[4, 0]
            # max_freq_nav = df_speed.iloc[5, 0]
            #
            # # Annotate the graph with navigating information
            # ax.text((avg_speed_nav - std_dev_nav), y_max * 0.5, 'Navigating', color='black')
            #
            # # Plot Gaussian distribution between 0 and 8 with .0001 steps.
            # x_axis = np.arange(0, 8, 0.0001)
            # ax.plot(x_axis, norm.pdf(x_axis, avg_speed_trawling, std_dev_trawling) * 200000, linestyle='-',
            #         color='blue')
            #
            # # Plot limits where we consider trawling speeds
            # plt.axvline(min_speed, color='blue', linestyle='--')
            # plt.axvline(max_speed, color='blue', linestyle='--')
        fig = ax.get_figure()
        fig.tight_layout()
    if fig_name is not None:
        with _print_with_time('Saving histogram as ' + fig_name + '.' + format_fig):
            file_name = fig_name + '.' + format_fig
            fig.savefig(file_name, format=format_fig, dpi=500)


# TODO Plot fishing effort raster