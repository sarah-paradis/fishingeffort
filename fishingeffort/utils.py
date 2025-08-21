import timeit
from contextlib import contextmanager
from datetime import datetime
import math
import geopandas as gpd
import cartopy.crs as ccrs


@contextmanager
def _print_with_time(*s):
    """
    Function to calculate the time it takes for a process to complete.
    To use, type "with print_with_time():"
    You should add in parenthesis the action you're doing as a string.
    The function prints out the action and the time it took to complete it in seconds
    """
    print(*s, end=f' [{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] ', flush=True)
    start = timeit.default_timer()
    yield
    print("\t[%.2fs]" % (timeit.default_timer() - start), flush=True)


def _UTM_zone_epsg(gdf):
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.crs.name == 'WGS 84' or isinstance(gdf.crs.utm_zone, str)

    if gdf.crs.name == 'WGS 84':
        # Step 1: Get the bounds (minx, miny, maxx, maxy)
        minx, miny, maxx, maxy = gdf.total_bounds

        # Step 2: Compute the centroid of the bounding box
        center_lon = (minx + maxx) / 2
        center_lat = (miny + maxy) / 2

        # Step 3: Compute UTM zone number
        utm_zone = math.floor((center_lon + 180) / 6) + 1

        # Step 4: Determine if it's in the northern or southern hemisphere
        hemisphere = 'north' if center_lat >= 0 else 'south'

        # Step 5: Build EPSG code
        if hemisphere == 'north':
            epsg_code = 32600 + utm_zone
        else:
            epsg_code = 32700 + utm_zone
        return epsg_code

    elif isinstance(gdf.crs.utm_zone, str):
        utm_str = gdf.crs.utm_zone
        utm_str = utm_str.strip().upper()
        hemisphere = utm_str[-1:]
        utm_zone = int(utm_str[:-1])
        if hemisphere == 'N':
            epsg_code = 32600 + utm_zone
        else:
            epsg_code = 32700 + utm_zone
        return epsg_code


def _UTM_zone(gdf):
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.crs.name == 'WGS 84' or isinstance(gdf.crs.utm_zone, str)

    if gdf.crs.name == 'WGS 84':
        # Step 1: Get the bounds (minx, miny, maxx, maxy)
        minx, miny, maxx, maxy = gdf.total_bounds

        # Step 2: Compute the centroid of the bounding box
        center_lon = (minx + maxx) / 2
        center_lat = (miny + maxy) / 2

        # Step 3: Compute UTM zone number
        utm_zone = math.floor((center_lon + 180) / 6) + 1

        # Step 4: Determine if it's in the northern or southern hemisphere
        hemisphere = 'N' if center_lat >= 0 else 'S'

        return utm_zone, hemisphere

    elif isinstance(gdf.crs.utm_zone, str):
        utm_str = gdf.crs.utm_zone
        utm_str = utm_str.strip().upper()
        hemisphere = utm_str[-1:]
        utm_zone = int(utm_str[:-1])
        return utm_zone, hemisphere


def _cartopy_UTM_projection(gdf):
    utm_zone, hemisphere = _UTM_zone(gdf)
    if hemisphere == 'S':
        southern_hemisphere = True
    else:
        southern_hemisphere = False
    return ccrs.UTM(utm_zone, southern_hemisphere=southern_hemisphere)

