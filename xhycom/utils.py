import csv
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import xarray as xr
from netCDF4 import Dataset, num2date, date2num
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import Stamen
from cartopy.io.img_tiles import OSM
from cartopy.mpl.ticker import LatitudeFormatter,LongitudeFormatter
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import hvplot.xarray
import warnings
import os
import sys
import datetime
from matplotlib.animation import FuncAnimation

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

def dims_from_opendap_dataset_access_form(year, opendap=False, dirpath="./gofs3.1/"):
    '''
    Reading catalog files of information about dimensions from remote or local
    In case of local, the files should be manually prepared using OPeNDAP Dataset Access Form
    by checking checkboxes for variables of depth, lat, lon, and time in a specified year
    http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2015.html
    and stored in dirpath.
    In case of remote, opendap core url without year should be spcified in dirpath like below:
    dirpath = "http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/"
    
    Args:
        year (str)  : target year
        opendap (bool) : True from opendap, False from local file
        dirpath(str): directory path for catalog fiels or opendap core url
    Returns:
       dimensions(dict): {"depth":depth, "lat":lat, "lon":lon, "time":time}
    '''

    if opendap:
        print('Accessing OPeNDAP catalog')
        infile = dirpath + year
        clg = Dataset(infile, mode='r')
        lon = clg.variables["lon"][:].data
        lat = clg.variables['lat'][:].data
        time = clg.variables['time'][:].data
        depth = clg.variables['depth'][:].data
        clg.close()
    else:
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
    Create netcdf for urls (multiple time) prepared by function set_urls.
    
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

def run_hycom_gofs3_1_region_ymdh(extent, time, opendap = False, dirpath = "./gofs3.1/", \
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
    dict_dims = dims_from_opendap_dataset_access_form(year=year, opendap=opendap, dirpath=dirpath)
    time, idx_time = gets_hycom_time(time_in_hours, array_time=dict_dims['time'])
    idx_lon_range = gets_index_range(lon_range, dict_dims['lon'])
    idx_lat_range = gets_index_range(lat_range, dict_dims['lat'])
    urls = set_urls(year, idx_lon_range, idx_lat_range, idx_time)
    ds = create_netcdf(urls)
    ### tau does not follow netCDF convention and not useful; deleted
    ds = ds.drop_vars("tau")

    ### Activate for pcolormesh()
    ### pcolormesh does not work even activating the following.
    #ds['time'] = pd.to_datetime(num2date(ds.time, units=ds.time.units, calendar="standard", \
    #                                     only_use_cftime_datetimes=False))
    #ds['time'].attrs["long_name"] = "Time"
    #ds['time'].attrs["axis"] = "T"
    #ds['time'].attrs["NAVO_code"]="13"
    ### End

    return ds

def run_opendap(extent, time_start, time_end=None, dtime=3, opendap=None, dirpath='./gofs3.1/', tz='utc'):
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
        str_region = "lon" + str(extent[0]) + "lon" + str(extent[1]) + "_lat" + str(extent[2]) + "lat" + str(extent[3])
        ncfile = "hycom_" + str_region + "_" + time.strftime('%Y-%m-%d_%H') + ".nc"
        if os.path.exists(ncfile):
            print(ncfile + ' exists and skip creating ' + ncfile + '.')
        else:
            ds = run_hycom_gofs3_1_region_ymdh(extent=extent, time=time_str, \
                 opendap=opendap, dirpath=dirpath)
            ds.to_netcdf(ncfile, mode="w")
            print("Creating " + ncfile + " complete.")


########################################################
# Class definitions
########################################################
class PlotConfig:
    '''
    Plotting configuration for 2D map
    '''

    def __init__(self, figsize=(6,5), tiler = Stamen('terrain-background'), zoom=9, cmap='magma_r', \
                 title_size=12, label_size=11, plot_coast='10m',\
                 proj='plate_carree', margins=(-0.1, 0.15, -0.1, 0.1), \
                 subplots_adjust=(0.1, 0.05, 0.95, 0.95, None, None), \
                 grid_linewidth=1, grid_linestyle=':', grid_color = 'k', grid_alpha=0.8, \
                 extend='both', cbar_kwargs={'shrink':0.9},title=None, \
                 vmin=None, vmax=None, levels=None):
        '''
        Instance method for class PlotConfig
        
        Args:
        figsize(tuple): (x_inches, y_inches)
        tiler(): tile or None
        zoom(int): Zoom for tile between 1 and 9 when tiler is specified.
        cmap(str): matplotlib colormap
        title_size(int): Font size for title or None for default
        label_size(int): Font size for label or None for default
        plot_coast(str): [ '10m' | '50m' | '110m' | None ] Plotting coastline
        proj(str): [ 'plate_carree' | None ] Set in case of lon-lat coordinates)
        margins(tuple): Margins in float for (L, B, R, T) in normalized coords between 0 and 1
        subplots_adjust(tuple): (L, B, R, T, wspace=None, hspace=None) for Figure.subplots_adjust()
        grid_linewidth(int|float): Grid line width
        grid_linestyle(str): Grid line style
        grid_color(str): Grid color
        grid_alpha(float): Grid alpha (opacity)
        cbar_kwargs(dict): kwargs for color bar 'shrink' adjusts the length of color bar
        extend(str): [ 'neither' | 'both' | 'min' | 'max' ] If not 'neither', 
                     make pointed end(s) for out-of- range values. 
                     These are set for a given colormap using the colormap set_under and set_over methods.
        vmin(float): vmin for cbar or None for default
        vmax(float): vmax for cbar or None for default
        levels(int | array-like): levels for cbar (int: num of levels, array-like: level values) or None for default
        '''

        self.figsize = figsize
        self.tiler = tiler
        self.zoom = zoom
        self.cmap = cmap
        self.title_size = title_size
        self.label_size = label_size
        ## self.ticks_intervals = ticks_intervals  ## ticks depends on plotting -> Plotter
        self.plot_coast = plot_coast
        self.proj = proj
        self.margins = margins
        self.subplots_adjust = subplots_adjust
        self.grid_linewidth = grid_linewidth
        self.grid_linestyle = grid_linestyle
        self.grid_color = grid_color
        self.grid_alpha = grid_alpha
        self.extend = extend
        self.cbar_kwargs=cbar_kwargs
        self.vmin=vmin
        self.vmax=vmax
        self.levels=levels
        self.title=title

class Data:
    '''
    Managing Xarray.DataArray in 4D(time, z, y, x) or 3D(time, y, x) prepared by reading netcdf.
    '''

    def __init__(self, da, vname=None, unit=None, xrange=None, yrange=None, zrange=None, trange=None, \
                 xlabel=None, ylabel=None, zlabel=None, tlabel=None):
        '''
        Instance method for class Data

        Args:
        da(Xarray.DataArray): Data to be managed
        vname(str): Name of variable. If None, gets from netcdf.
        unit(str) : Name of unit. If None, gets from netcdf.
        xrange(tuple): Range of x (lon)   dimension. If None, gets from netcdf.
        yrange(tuple): Range of y (lat)   dimension. If None, gets from netcdf.
        zrange(tuple): Range of z (depth) dimension. If None, gets from netcdf. Only in 4D data
        trange(tuple): Range of t (time)  dimension. If None, gets from netcdf. 
        xlabel(str)  : Label of x. If None, gets from netcdf.
        ylabel(str)  : Label of y. If None, gets from netcdf.
        zlabel(str)  : Label of z. If None, gets from netcdf. Only in 4D data
        tlabel(str)  : Label of t. If None, gets from netcdf.
        
        '''
        
        self.da = da                      ## xarray.DataArray
        self.vmax = self.da.max().values
        self.vmin = self.da.min().values
        if vname is None:
            self.vname = self.da.long_name
        else:
            self.vname = vname
        if unit is None:
            self.unit = self.da.units
        else:
            self.unit = unit

        ### 3D and 4D
        if xrange is None:
            self.xrange = (self.da[self.da.dims[-1]].values.min(), self.da[self.da.dims[-1]].values.max())
        else:
            self.xrange = xrange
        self.xmin, self.xmax = self.xrange
        if yrange is None:
            self.yrange = (self.da[self.da.dims[-2]].values.min(), self.da[self.da.dims[-2]].values.max())
        else:
            self.yrange = yrange
        self.ymin, self.ymax = self.yrange
        if xlabel is None:
            self.xlabel = self.da.dims[-1]
        else:
            self.xlabel = xlable
        if ylabel is None:
            self.ylabel = self.da.dims[-2]
        else:
            self.ylabel = ylable

        ### 4D (t, z, y, x)
        if len(self.da.dims) == 4:
            if zrange is None:
                self.zrange = (self.da[self.da.dims[-3]].values.min(), self.da[self.da.dims[-3]].values.max())
            else:
                self.zrange = zrange
            self.zmin, self.zmax = self.zrange
            if zlabel is None:
                self.zlabel = self.da.dims[-3]
            else:
                self.zlabel = zlable
        ### Time
        if trange is None:
            self.trange = (self.da[self.da.dims[0]].values.min(), self.da[self.da.dims[0]].values.max())
        else:
            self.trange = trange
        self.tmin, self.tmax = self.trange
        if tlabel is None:
            self.tlabel = self.da.dims[0]
        else:
            self.tlabel = tlabel
                
class Plotter:
    '''
    Plotting 2D maps (top view or vertical view) invoking the instances of PlotConfig and Data
    '''
    def __init__(self, plot_config, data, x='lon', y='lat', z='depth', t='time'):
        '''
        Instance method for Plotter. Type of plot is determined by args of x, y, z, and t.
        For example, if x='lon', y='lat', z=0, and t=0, a horizontal x-y plot is selected
        with indexing of z=0 and t=0.
        When t=slice(None), slicing all t values, used for creating animation.

        Args:
        plot_config(PlotConfig): Gets information about configuration
        data(Data): Gets data and its specification.
        x(str | int): Name or index of of x dimension
        y(str | int): Name or index of of y dimension
        z(str | int): Name or index of of z dimension
        t(str | int): Name or index of of t dimension
        '''
 
        self.cfg = plot_config
        self.data = data
        ### Prepare projection for (lon, lat) coordinates
        if self.cfg.proj == 'plate_carree':
            self.proj = ccrs.PlateCarree()
        else:
            self.proj = ccrs.PlateCarree()  ## Default
        self.x = x
        self.y = y
        self.z = z
        self.t = t
        ### Determine the type of plot and projection
        if self.x   == 'lon'  and self.y == 'lat':
            self.plot_type = 'xy_view'
            self.indexing = dict(time=self.t, depth=self.z)
            self.x_axis = self.x
            self.y_axis = self.y
        else:
            self.proj = None
        if self.x == 'lon'  and self.z == 'depth':
            self.plot_type = 'xz_view'
            self.indexing = dict(time=self.t, lat = self.y)
            self.x_axis=self.x
            self.y_axis=self.z
        elif self.y == 'lat'  and self.z == 'depth':
            self.plot_type = 'yz_view'
            self.indexing = dict(time=self.t, lon = self.x)
            self.x_axis = self.y
            self.y_axis = self.z
        elif self.t == 'time' and self.x == 'lon':
            self.plot_type = 'tx_view'
            self.indexing = dict(depth = self.z, lat = self.y)
            self.x_axis = self.t
            self.y_axis = self.x
        elif self.t == 'time' and self.y == 'lat':
            self.plot_type = 'ty_view'
            self.indexing = dict(depth = self.z, lon = self.x)
            self.x_axis = self.t
            self.y_axis = self.y
        elif self.t == 'time' and self.z == 'depth':
            self.plot_type = 'tz_view'
            self.indexing = dict(lat = self.y, lon = self.x)
            self.x_axis = self.t
            self.y_axis = self.z
        
        self.fig = plt.figure(figsize=self.cfg.figsize)
        
        
    def get_ax(self):
        '''
        Create and return Axes.
        '''

        print("projection = ", self.proj)
        if self.proj is None:
            return self.fig.add_subplot(1,1,1)
        else:
            return self.fig.add_subplot(1,1,1, projection=self.proj)

    def update_ax(self, ax):
        '''
        Update Axes.
        '''

        plt.rcParams['font.size'] = self.cfg.title_size

        if self.plot_type == 'xy_view':
            ax.set_extent(self.extent, crs=self.proj)
            if self.cfg.tiler is None:
                ax.coastlines()
                gl=ax.gridlines(draw_labels=True, xlocs=self.xticks, ylocs=self.yticks, \
                                linestyle=self.cfg.grid_linestyle, linewidth=self.cfg.grid_linewidth, \
                                color=self.cfg.grid_color, alpha=self.cfg.grid_alpha)
                gl.right_labels=False
                gl.top_labels=False
                gl.xlabel_style={'size':self.cfg.label_size}
                gl.ylabel_style={'size':self.cfg.label_size}
                if self.xticks is not None:
                    gl.xlocator = mticker.FixedLocator(self.xticks)
                if self.yticks is not None:
                    gl.ylocator = mticker.FixedLocator(self.yticks)
            else:
                gl=ax.gridlines(crs=self.proj, draw_labels=False, linestyle=self.cfg.grid_linestyle, \
                                linewidth=self.cfg.grid_linewidth, color=self.cfg.grid_color,\
                                alpha=self.cfg.grid_alpha)
                if self.xticks is not None:
                    gl.xlocator = mticker.FixedLocator(self.xticks)
                    ax.set_xticks(self.xticks,crs=self.proj)
                if self.yticks is not None:
                    gl.ylocator = mticker.FixedLocator(self.yticks)
                    ax.set_yticks(self.yticks,crs=self.proj)
                if self.proj is not None:
                    latfmt=LatitudeFormatter()
                    lonfmt=LongitudeFormatter(zero_direction_label=True)
                    ax.xaxis.set_major_formatter(lonfmt)
                    ax.yaxis.set_major_formatter(latfmt)
                ax.axes.tick_params(labelsize=self.cfg.label_size)
                ax.add_image(self.cfg.tiler, self.cfg.zoom)
        else:
            if self.plot_type == 'tx_view' or self.plot_type == 'ty_view' or \
               self.plot_type == 'tz_view':
                ### set_xlim() does not work for time series axis
                #ax.set_xlim(pd.to_datetime(self.extent[0]).to_pydatetime(), \
                #            pd.to_datetime(self.extent[1]).to_pydatetime())
                print('Notice: For time series axis, set_xlim() does not work.')
            else:
                ax.set_xlim(self.extent[0], self.extent[1])
            ax.set_ylim(self.extent[2], self.extent[3])
            ax.grid(linestyle=self.cfg.grid_linestyle, linewidth=self.cfg.grid_linewidth, \
                    color=self.cfg.grid_color, alpha=self.cfg.grid_alpha)
            if self.xticks is not None:
                ax.set_xticks(self.xticks)
            if self.yticks is not None:
                ax.set_yticks(self.yticks)
            ax.tick_params(labelsize=self.cfg.label_size)
            if self.plot_type == 'xz_view' or self.plot_type == 'yz_view' or \
               self.plot_type == 'tz_view':
                ax.invert_yaxis()  ## reverse y-axis when y axis is depth.
        
        return ax

    def update_title(self, ax):
        '''
        Updating title using ax.set_title()
        '''

        title = self.cfg.title
        if title == None:
            if self.plot_type == 'xy_view':
                title = str(self.data.da['time'].values[self.t])[0:16] + \
                                 "  Depth = " + str(self.data.da['depth'].values[self.z])[0:5]
            elif self.plot_type == 'xz_view':
                title = str(self.data.da['time'].values[self.t])[0:16] + \
                                 "  Lat = " + str(self.data.da['lat'].values[self.y])[0:5]
            elif self.plot_type == 'tx_view':
                title = "Lat = " + str(self.data.da['lat'].values[self.y])[0:5] + \
                             "  Depth = " + str(self.data.da['depth'].values[self.z])[0:5]
            elif self.plot_type == 'ty_view':           
                title = "Lon = " + str(self.data.da['lon'].values[self.x])[0:5] + \
                        "  Depth = " + str(self.data.da['depth'].values[self.z])[0:5]
            elif self.plot_type == 'tz_view':
                title = "Lon = " + str(self.data.da['lon'].values[self.x])[0:5] + \
                        "  Lat = " + str(self.data.da['lat'].values[self.y])[0:5]
        ax.set_title(title)

        return ax

    def make_2d_plot(self, ax, **kwargs):
        '''
        Update Axes by plotting 2D horizontal, vertical, or time series panel.
        Default da.plot() does not work for time series probably because of unsupported datetime64;
        thus it is replaced with da.plot.contourf(). In addition, ax.set_xlim() cannot be applied.
        '''

        if self.plot_type == 'xy_view':  ### Projection required
            self.data.da[self.indexing].plot(ax=ax, x=self.x_axis, y=self.y_axis, \
                    cmap=self.cfg.cmap, transform=self.proj, extend=self.cfg.extend, \
                    vmin=self.cfg.vmin, vmax=self.cfg.vmax, \
                    levels=self.cfg.levels, cbar_kwargs=self.cfg.cbar_kwargs, **kwargs)
        else:  ### No projection
            if self.plot_type == 'xz_view' or self.plot_type == 'yz_view':
                self.data.da[self.indexing].plot(ax=ax, x=self.x_axis, y=self.y_axis, \
                extend=self.cfg.extend, vmin=self.cfg.vmin, vmax=self.cfg.vmax, cmap=self.cfg.cmap, \
                levels=self.cfg.levels, cbar_kwargs=self.cfg.cbar_kwargs, **kwargs)
            elif self.plot_type == 'tx_view' or self.plot_type == 'ty_view' or\
                 self.plot_type == 'tz_view':  ### contourf() required
                self.data.da[self.indexing].plot.contourf(ax=ax, x=self.x_axis, y=self.y_axis, \
                extend=self.cfg.extend, vmin=self.cfg.vmin, vmax=self.cfg.vmax, cmap=self.cfg.cmap, \
                levels=self.cfg.levels, cbar_kwargs=self.cfg.cbar_kwargs, **kwargs)
            else:
                print('ERROR: No such plot type of ', self.plot_type)

        return ax

    def _set_plot_extent(self, extent, ticks_intervals, vmin, vmax, levels):
        '''
        Private method invoked by plot, save, and anim methods for setting plot extent, ticks intervals,
        variable range of vmin and vmax and contour levels.
        '''

        self.extent = extent
        self.ticks_intervals = ticks_intervals
        if self.extent is None:
            if self.plot_type == 'xy_view':
                self.extent = (self.data.xmin, self.data.xmax, self.data.ymin, self.data.ymax)
            elif self.plot_type == 'xz_view':
                self.extent = (self.data.xmin, self.data.xmax, self.data.zmin, self.data.zmax)
            elif self.plot_type == 'yz_view':
                self.extent = (self.data.ymin, self.data.ymax, self.data.zmin, self.data.zmax)
            elif self.plot_type == 'tx_view':
                self.extent = (self.data.tmin, self.data.tmax, self.data.xmin, self.data.xmax)
            elif self.plot_type == 'ty_view':
                self.extent = (self.data.tmin, self.data.tmax, self.data.ymin, self.data.ymax)
            elif self.plot_type == 'tz_view':
                self.extent = (self.data.tmin, self.data.tmax, self.data.zmin, self.data.zmax)
            else:
                print('Error: such plot_type is not defined in set_plot_extent')
        self.xmin, self.xmax = self.extent[0:2]
        self.ymin, self.ymax = self.extent[2:4]   

        if self.ticks_intervals is None:
            self.xticks = None
            self.yticks = None
        else:
            self.xticks = np.arange(self.xmin, self.xmax, self.ticks_intervals[0])
            self.yticks = np.arange(self.ymin, self.ymax, self.ticks_intervals[1])

        if vmin is not None:
            self.cfg.vmin = vmin  ### Override
        if vmax is not None:
            self.cfg.vmax = vmax  ### Override
        if levels is not None:
            self.cfg.levels = levels  ### Override

    def _plot(self, **kwargs):
        '''
        Private instance invoked by save method.
        '''
        ax = self.get_ax()
        ax = self.make_2d_plot(ax, **kwargs)
        ax = self.update_title(ax)
        ax = self.update_ax(ax)  ## Needs to invoke at the last
        return ax

    def plot(self, extent=None, ticks_intervals=None, vmin=None, vmax=None, levels=None, **kwargs):
        '''
        Public method for plotting on a screen. Arguments are to override parameter
        values set in PlotConfig.
        
        Args:
        extent(tuple)           : Extent of plot. Default is None which gets from PlotConfig.
        ticks_intervals(tuple)  : Default is None which gets from PlotConfig.
        vmin(float)             : Min variable value. Default is None which gets from PlotConfig.
        vmax(float)             : Max variable value. Default is None which gets from PlotConfig.
        levels(int | array-like): Number of levels in int or list of levels.
                                  Default is None which gets from PlotConfig.
        '''

        self._set_plot_extent(extent, ticks_intervals, vmin, vmax, levels)
        self._plot(**kwargs)
        return self

    def save(self, filename, **kwargs):
        '''
        Public method of plot method (Plotter().plot().save()) for creating graphic file.

        Args:
        filename(str): File name with an extension of graphic format, e.g., png.
        **kwargs     : kwargs for Figure.savefig()
        '''
        
        self.fig.savefig(filename, **kwargs)
        print("Saved {}.".format(filename))

    def frame(self, extent=None, ticks_intervals=None, vmin=None, vmax=None, levels=None, \
              subplot_adjust=(0.15, 0.05, 0.9, 0.95), **kwargs):
        '''
        Creating an initial frame of animation.
        
        Args:
        extent(tuple)           : Extent of plot. Default is None which gets from PlotConfig.
        ticks_intervals(tuple)  : Default is None which gets from PlotConfig.
        vmin(float)             : Min variable value. Default is None which gets from PlotConfig.
        vmax(float)             : Max variable value. Default is None which gets from PlotConfig.
        levels(int | array-like): Number of levels in int or list of levels.
                                  Default is None which gets from PlotConfig.
        subplot_adjust(tuple)   : Tuple of (left, bottom, right, top) in normalized coords.
        **kwargs                : DataArray.plot(**kwargs)
        '''

        left = subplot_adjust[0]
        bottom = subplot_adjust[1]
        right = subplot_adjust[2]
        top = subplot_adjust[3]

        self.fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=None, hspace=None)
        self._set_plot_extent(extent, ticks_intervals, vmin, vmax, levels)
        ax = self.get_ax()
        ax = self.update_ax(ax)
        self.cax = self.data.da.isel(depth=self.z, time=0).plot(ax=ax, x='lon', y='lat', transform=self.proj,\
                                                          cmap=self.cfg.cmap, extend=self.cfg.extend, \
                                                          vmin=vmin, vmax=vmax, levels=levels, \
                                                          **kwargs)
        #ani = FuncAnimation(self.fig, anim_update, frames=3, interval=200, blit=False, repeat=True)
        #ani.save(filename, **kwargs)
        self.ax_anim = ax
        return self


    def anim(self, filename, frames=None, interval=200, blit=False, repeat=True, **kwargs):
        '''
        Method of medhod frame for creating GIF animation. There is a kind of bug for plotting an array
        including nan with set_array() method which is required when creating animation;
        thus masked_array needs to be used instead of ndarray.
        https://stackoverflow.com/questions/58117358/matplotlib-image-plot-nan-values-shown-as-lowest-color-of-colormaps-instead-of
        http://xarray.pydata.org/en/stable/generated/xarray.DataArray.to_masked_array.html
        
        Args:
        filename(str): GIF or MP4 file name.
        frames(int)  : Number of frames
        interval(int): Interval in ms
        blit
        repeat(bool) : True: repeat
        **kwargs                : kwargs for Animation.save
        '''

        def anim_update(i):
            masked_array = self.data.da.isel(depth=self.z, time=i).to_masked_array()
            self.cax.set_array(masked_array.flatten())
            self.ax_anim.set_title("Depth = " + str(self.data.da['depth'].values[self.z]) + \
                         "  Time = " + str(self.data.da.coords['time'].values[i])[0:16])

        print(self.data.da['time'][self.t])
        if frames is None:
            frames = len(self.data.da['time'][self.t])
        ani = FuncAnimation(self.fig, anim_update, frames=frames, interval=interval, blit=blit, repeat=repeat)
        ani.save(filename, **kwargs)
        return ani


########################################################
if __name__ == "__main__":
########################################################
    extent = (139.5, 140.0, 34.8, 35.2)
    time = '2012-10-28 12:00:00'
    ds = run_hycom_gofs3_1_region_ymdh(extent=extent, time=time)
    print(ds)
