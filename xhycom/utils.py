import pandas as pd
import numpy as np
import csv
from datetime import datetime, timedelta, timezone
from dateutil.parser import parse
import pytz
import sys
import netCDF4
from netCDF4 import num2date
import subprocess
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import Stamen
from cartopy.io.img_tiles import OSM
from cartopy.mpl.ticker import LatitudeFormatter,LongitudeFormatter
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# latまたはlonの範囲を入力し，それを囲む最小の領域のインデックスを返す
def gets_index_range(degree_range, dims_opendap_form_val):
    '''Function to get indexes of nearest outside degree range of val

    Args:
        degree_range (tuple): range of region in degree for lat or lon
        dims_opendap_form_val (numpy.ndarray()): Opendap data dimension array
            for lon or lat, whose index numbers (idx_min and idx_max)
            corresponding to the min and max of degree_range are to be determined.
    Returns:
        tuple: Index of min (idx_min) and max (idx_max) of opendap data dimension
    '''

    idx_min = np.abs(degree_range[0] - dims_opendap_form_val).argmin()
    idx_max = np.abs(degree_range[1] - dims_opendap_form_val).argmin()
    #print(idx_min, idx_max)
    if dims_opendap_form_val[idx_min] > degree_range[0]:
        idx_min -= 1
    if dims_opendap_form_val[idx_max] < degree_range[1]:
        idx_max += 1
    return idx_min, idx_max

# ymdhを入力し，基準からの時間数を実数または一番近い整数で返す
# 基準はopendap dataやmodelの仕様に合わせる必要がある
def gets_hours_from_time_origin(time, time_org='2000-01-01 00:00:00', tz='utc', \
                                tz_diff=0, val_int=True):
    '''Gets cumulative hours at a specified time from a specified time origin
       https://qiita.com/shota243/items/91660ece72b5e84c3adb
       Should be considered in utc and converting t local time if necessary

       Args:
           time (str): specified time  (e.g., "2012-01-01 12:00:00")
           time_org (str): time origin in utc
           tz (int): time zone (default: utc)
           tz_diff (int): time zone difference (e.g., -9 for jst)
           val_int (bool): Return value is integer when True or real when False 
       Returns:
           int/real, pandas.Timestamp: number of hours from the time origin (int or real) and
               specified time in Timestamp 
    '''

    time_org = pd.Timestamp(time_org).tz_localize('utc')
    time_stamp = pd.Timestamp(time).tz_localize(tz) + timedelta(hours=1) * tz_diff
    print('Target', time_stamp)
    # time_org = datetime.fromisoformat('2000-01-01T00:00:00')
    print('Origin', time_org)
    num_hours = (time_stamp - time_org).total_seconds()/3600.0
    if val_int == True:
        num_hours = int(num_hours)
    return num_hours, time_stamp

def dims_from_opendap_dataset_access_form(year, dirpath="./gofs3.1/"):
    '''
    Reading catalog files of information about dimensions
    The files should be manually prepared using OPeNDAP Dataset Access Form
    by checking checkboxes for variables of depth, lat, lon, and time in a specified year
    http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2015.html
    and stored in dirpath.
    
    Args:
        year (str)  : target year
        dirpath(str): directory path for catalog fiels
    Returns:
       dimensions(dict): {"depth":depth, "lat":lat, "lon":lon, "time":time}
    '''

    infile = dirpath + year + ".dpb.txt"
    with open(infile) as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
    depth = np.array(rows[8], dtype='float')
    lat   = np.array(rows[11], dtype='float')
    lon   = np.array(rows[14], dtype='float')
    time  = np.array(rows[17], dtype='float')
    time  = time.astype(np.int)
    dict_dims = {"depth": depth, "lat": lat, "lon": lon, "time": time}
    return dict_dims

def gets_hycom_time(time_in_hours, array_time):
    '''Gets closest hycom time idx correspoding to a given time
    
    Args:
        time_in_hours (int/real): time in num_hours (see function gets_hours_from_time_origin)
        array_time (array): Array of hycom time dimension obtained by
        function of dims_from_opendap_dataset_access_form
    Returns:
        tuple: Closest hycom time and its index
    '''

    idx = np.abs(array_time - time_in_hours).argmin()
    if np.abs(time_in_hours - array_time[idx]) > 1:
        print("ERROR: Specified time of ", time_in_hours, "is out of range between", array_time[0], "and", array_time[-1])
        sys.exit("ERROR: Spedified time is out of range.")
    #print(idx)
    return array_time[idx], idx

# 年と空間領域と時間のインデックスを入力し，HYCOMデータダウンロードurlのリストを返す
# 1度にダウンロード可能なデータ量に制約があるためか，urlを分割しないとエラーとなる
def set_urls(year, idx_lon_range, idx_lat_range, idx_time):
    '''
    Set urls for Hycom data download by specifying indexes of regional range and time.

    Args:
        year (str):
        idx_lon_range (tuple): idx_lon_min, idx_lon_max
        idx_lat_range (tuple): idx_lat_min, idx_lat_max
        idx_time (int):
    Returns:
        list(str): List of urls for HYCOM OPeNDAP download
    '''

    idx_lon_min, idx_lon_max = idx_lon_range
    idx_lat_min, idx_lat_max = idx_lat_range
    txt_core = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/{}?'.format(year)
    txt_dims = 'depth[0:1:39],lat[{}:1:{}],lon[{}:1:{}],time[{}:1:{}],tau[{}:1:{}]'\
            .format(idx_lat_min, idx_lat_max, idx_lon_min, idx_lon_max,idx_time,\
                    idx_time, idx_time, idx_time)
    #print(txt_dims)
    txt_water_u = ',water_u[{}:1:{}][0:1:39][{}:1:{}][{}:1:{}]'\
               .format(idx_time, idx_time, idx_lat_min, idx_lat_max, idx_lon_min, idx_lon_max)
    txt_water_u_bottom = ',water_u_bottom[{}:1:{}][{}:1:{}][{}:1:{}]'\
               .format(idx_time, idx_time, idx_lat_min, idx_lat_max, idx_lon_min, idx_lon_max)
    #print(txt_water_u_bottom)
    txt_water_v = ',water_v[{}:1:{}][0:1:39][{}:1:{}][{}:1:{}]'\
               .format(idx_time, idx_time, idx_lat_min, idx_lat_max, idx_lon_min, idx_lon_max)
    #print(txt_water_v)
    txt_water_v_bottom = ',water_v_bottom[{}:1:{}][{}:1:{}][{}:1:{}]'\
               .format(idx_time, idx_time, idx_lat_min, idx_lat_max, idx_lon_min, idx_lon_max)
    txt_water_temp = ',water_temp[{}:1:{}][0:1:39][{}:1:{}][{}:1:{}]'\
           .format(idx_time, idx_time, idx_lat_min, idx_lat_max, idx_lon_min, idx_lon_max)
    #print(txt_water_temp)
    txt_water_temp_bottom = ',water_temp_bottom[{}:1:{}][{}:1:{}][{}:1:{}]'\
               .format(idx_time, idx_time, idx_lat_min, idx_lat_max, idx_lon_min, idx_lon_max)
    txt_salt = ',salinity[{}:1:{}][0:1:39][{}:1:{}][{}:1:{}]'\
               .format(idx_time, idx_time, idx_lat_min, idx_lat_max, idx_lon_min, idx_lon_max)
    #print(txt_salt)
    txt_salt_bottom = ',salinity_bottom[{}:1:{}][{}:1:{}][{}:1:{}]'\
               .format(idx_time, idx_time, idx_lat_min, idx_lat_max, idx_lon_min, idx_lon_max)
    txt_surf_el = ',surf_el[{}:1:{}][{}:1:{}][{}:1:{}]'\
               .format(idx_time, idx_time, idx_lat_min, idx_lat_max, idx_lon_min, idx_lon_max)

    urls = []
    urls.append(txt_core + txt_dims + txt_water_u + txt_water_u_bottom)
    urls.append(txt_core + txt_dims + txt_water_v + txt_water_v_bottom)
    urls.append(txt_core + txt_dims + txt_water_temp + txt_water_temp_bottom)
    urls.append(txt_core + txt_dims + txt_salt + txt_salt_bottom + txt_surf_el)

    return urls

def create_netcdf(urls):
    '''
    Create netcdf for urls prepared by function set_urls.
    
    Args:
        urls(list): Urls for HYCOM OPeNDAP
    Returns:
        xarray.Dataset:
    '''

    ### In case decoding fails, activate "decode_times=False"
    ds =  xr.open_dataset(urls[0], decode_times=False)
    ### ds =  xr.open_dataset(urls[0])
    
    glb_attrs = ds.attrs
    ncfiles=[ds]
    ncfiles.extend([xr.open_dataset(urls[i], decode_times=False) for i in range(1,4)])
    #print(ncfiles)
    DS = xr.merge(ncfiles)
    DS.attrs = glb_attrs
    return DS


def time_range(time_start, time_end, dtime):
    '''
    Generator of Timestamp from time_start to time_end with dtime interval

    Args:
        time_start (pandas.Timestamp): Start time (date and time)
        time_end (pandas.Timestamp): End time (date and time)
        dtime (pandas.Timedelta): time interval
    Yields:
        Timestamp: The next time between time_start and time_end
    '''

    if time_start > time_end:
        sys.exit("ERROR: time_start > time_end")
    while True:
        if time_start <= time_end:
            yield time_start
            time_start += dtime
        else:
            break


def ax_lonlat_axes(ax, extent, grid_linestyle=':', grid_linewidth=0.5, grid_color='k', \
                   grid_alpha=0.8, xticks=None, yticks=None, label_size=12, tiler=None, zoom=8):
    '''
    Modify Axis in matplotlib

    Args:
        ax(Axis)                 : gets current Axis
        extent (tuple)           : (lon_min, lon_max, lat_min, lat_max)
        grid_linestyle(str)      : linestyle (default: ':')
        grid_linewidth(float/int): linewidth (default: 0.5)
        grid_color(str)          : color (default: 'k')
        grid_alpha(float)        : opacity (default: 0.8)
        xticks(list)             : list of xticks (default: None)
        yticks(list)             : list of yticks (default: None)
        label_size(int)          : axes label size in pt (default: 12)
        tiler(cartopy.io.img_tiles):
        zoom(int)                : zoom in tiler
    Returns:
        ax(Axis)                 : overrides Axis
        
    '''
    lon_min, lon_max = extent[0:2]
    lat_min, lat_max = extent[2:4]   
    ax.set_extent([lon_min,lon_max,lat_min,lat_max], crs=ccrs.PlateCarree())
    if tiler is None:
        ax.coastlines()
        gl=ax.gridlines(draw_labels=True, xlocs=xticks, ylocs=yticks, linestyle=grid_linestyle, \
                        linewidth=grid_linewidth, color=grid_color, alpha=grid_alpha)
        gl.right_labels=False
        gl.top_labels=False
        gl.xlabel_style={'size':label_size}
        gl.ylabel_style={'size':label_size}
        if xticks is not None:
            gl.xlocator = mticker.FixedLocator(xticks)
        if yticks is not None:
            gl.ylocator = mticker.FixedLocator(yticks)
    else:
        #plt.rcParams['font.size'] = label_size
        gl=ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linestyle=grid_linestyle, \
                        linewidth=grid_linewidth, color=grid_color,alpha=grid_alpha)
        gl.xlocator = mticker.FixedLocator(xticks)
        gl.ylocator = mticker.FixedLocator(yticks)
        ax.set_xticks(xticks,crs=ccrs.PlateCarree())
        ax.set_yticks(yticks,crs=ccrs.PlateCarree())
        latfmt=LatitudeFormatter()
        lonfmt=LongitudeFormatter(zero_direction_label=True)
        ax.xaxis.set_major_formatter(lonfmt)
        ax.yaxis.set_major_formatter(latfmt)
        ax.axes.tick_params(labelsize=label_size)
        ax.add_image(tiler, zoom)
    #plt.rcParams['font.size'] = label_size
    return ax


# Run functions

def run_hycom_gofs3_1_region_ymdh(extent, time, \
                                  time_org='2000-01-01 00:00:00', tz='utc'):
    '''
    Downloding HYCOM dataset using OPeNDAP by specifying spatial region and time
    and create xarray.Dataset

    Args:
       extent (tuple): (lon_min, lon_max, lat_min, lat_max)
       time (str): time in 'YYYY-MM-DD HH:mm:SS'
       time_org (str): Origin of time in 'YYYY-MM-DD HH:mm:SS'
       tz (str): time zone
    Returns:
       (xarray.Dataset)
       
    '''

    lon_range, lat_range = extent[0:2], extent[2:4]
    time_in_hours, time_ymdh = gets_hours_from_time_origin(time=time, time_org=time_org, \
                                                          tz=tz, tz_diff=0, val_int=True)
    print(time_in_hours, time_ymdh)
    year = time_ymdh.strftime('%Y')
    dict_dims = dims_from_opendap_dataset_access_form(year)
    time, idx_time = gets_hycom_time(time_in_hours, array_time=dict_dims['time'])
    idx_lon_range = gets_index_range(lon_range, dict_dims['lon'])
    idx_lat_range = gets_index_range(lat_range, dict_dims['lat'])
    urls = set_urls(year, idx_lon_range, idx_lat_range, idx_time)
    ds = create_netcdf(urls)
    ### tau does not follow netCDF convention and not useful; deleted
    ds = ds.drop_vars("tau")
    ### Modify time coord
    #ds['time'] = pd.to_datetime(num2date(ds.time, units=ds.time.units, calendar="standard", \
    #                                     only_use_cftime_datetimes=False))
    ds['time'].attrs["long_name"] = "Time"
    #ds['time'].attrs["axis"] = "T"
    #ds['time'].attrs["NAVO_code"]="13"
    return ds

def run_opendap(extent, time_start, time_end=None, dtime=3, tz='utc'):
    '''
    Download multiple OPeNDAP files for a specific region and time period
    using run_hycom_gofs3_1_region_ymdh().
    
    Args:
        extent (tuple)   : (lon_min, lon_max, lat_min, lat_max)
        time_start (str) : start datetime
        time_end (str)   : end datetime
        tz(str)          : time zone (default: 'utc')
        dtime (int)      : time interval in hours (default: 3)
    Returns:
        None: Creating netcdf file for each time
    '''

    if time_end is None:
        time_end = time_start
    dtime = dtime * timedelta(hours=1)
    time_start = pd.Timestamp(time_start).tz_localize(tz)
    time_end =   pd.Timestamp(time_end).tz_localize(tz)
    
    for time in time_range(time_start, time_end, dtime):
        ### arg type of time of run_hycom_gofs3_1_region_ymdh is str.
        time_str = time.strftime('%Y-%m-%d %H:%M:%S')
        ### ncfile name cannot contain ':' or ' '.
        ncfile = "hycom_" + time.strftime('%Y-%m-%d_%H') + ".nc"
        ds = run_hycom_gofs3_1_region_ymdh(extent=extent, time=time_str)
        ds.to_netcdf(ncfile, mode="w")
        print("Creating " + ncfile + " complete.")


########################################################
if __name__ == "__main__":
########################################################
    extent = (139.5, 140.0, 34.8, 35.2)
    time = '2012-10-28 12:00:00'
    ds = run_hycom_gofs3_1_region_ymdh(extent=extent, time=time)
    print(ds)
