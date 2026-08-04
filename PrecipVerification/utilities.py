import argparse
import calendar
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime as dt
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import subprocess
import sys
import time as time_module
import utilities as utils
import warnings
import xarray as xr
from netCDF4 import Dataset

# Communicate command output using subprocess
def comm_cmd(cmd_str):
    cmd_out = subprocess.Popen(cmd_str, stdout=subprocess.PIPE, shell = True).communicate()[0].strip().decode()
    return cmd_out

if (comm_cmd("echo $CONDA_DEFAULT_ENV") == "xesmf_env"):
    import xesmf
    import cf_xarray as cfxr

# Suppress all warnings like "More than 20 figures have been opened"
# and "All-NaN slice encountered" in percentile calculations
def suppress_warnings(): 
    warnings.filterwarnings("ignore")

# STANDARD/COMMON DIMENSION AND VARIABLE NAMES 
#################################################################################
# Dimension names
time_dim_str = "time"
period_begin_time_dim_str = "period_begin_time"
period_end_time_dim_str = "period_end_time"
days_dim_str = "days"
months_dim_str = "months"
seasons_dim_str = "seasons"
annual_dim_str = "years"
full_period_dim_str = "full_period"
common_month_dim_str = "common_month"
common_season_dim_str = "common_season"

# Variable names
accum_precip_var_name = "accum_precip"

# Common date string formats
full_date_format_str = "%Y-%m-%d %H:%M:%S"

# Reference time for Unix time (i.e., a date in expressed in unix seconds will be seconds from this date
UNIX_EPOCH = dt.datetime(1970, 1, 1, 0, 0, 0)
unix_epoch_str = UNIX_EPOCH.strftime(full_date_format_str) # UNIX_EPOCH in format "1970-01-01 00:00:00"
seconds_since_unix_epoch_str = f"seconds since {unix_epoch_str}"

# Replay native grid cell size in degrees
replay_grid_cell_size = 0.234375

# AORC native grid cell size in degrees
aorc_grid_cell_size = 0.033332

# Nested Eagle ~15km grid cell size
nested_eagle_mean_lon_step = 0.16802844
nested_eagle_mean_lat_step = 0.13105997

# Nested Eagle ~06km grid cell size
nested_eagle_mean_lon_step_hi_res = 0.06739469
nested_eagle_mean_lat_step_hi_res = 0.05261715

# Take the mean of the lat and lon steps, then subsequently the mean of those values
nested_eagle_grid_cell_size = 0.5 * (nested_eagle_mean_lon_step + nested_eagle_mean_lat_step)
nested_eagle_grid_cell_size_hi_res = 0.5 * (nested_eagle_mean_lon_step_hi_res + nested_eagle_mean_lat_step_hi_res)

# Radius, ARIs (in years), amount, and percentile thresholds for various
# types of evaluations; typically used for FSS and occurence stats (like frequency bias)

# Default list of evaluation radii (number of grid cells)
default_eval_radius_list_grid_cells = np.array([1, 2, 3, 4, 6, 8, 12, 20, 40, 80])

# Default list of evaluation radii (degrees latitude/longitude, for a 0.25 degree grid) 
#default_eval_radius_list_deg = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0])
default_eval_radius_list_deg = np.copy(default_eval_radius_list_grid_cells) * 0.25

# Default list of evaluation thresholds (mm)
default_eval_threshold_list_mm = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 75.0, 80.0, 90.0, 100.0])

# Default list of percentile evaluation thresholds
default_eval_threshold_list_pctl = np.array([5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0, 99.9])

# Default list of ARI grids to use as thresholds (years) 
default_eval_ari_list_years = np.array([1, 2, 5, 10, 25, 50, 100, 200, 500, 1000])

AORC_data_name = "AORC"
CONUS404_data_name = "CONUS404"
ERA5_data_name = "ERA5"
HRRR_data_name = "HRRR"
IMERG_data_name = "IMERG"
NestedReplay_data_name = "NestedReplay"
Replay_data_name = "Replay"

dataset_list = [AORC_data_name, CONUS404_data_name, ERA5_data_name, HRRR_data_name,
                IMERG_data_name, NestedReplay_data_name, Replay_data_name]

# Convert 'period_end_time' dimension name and associated coordinates to 'period_begin_time'
# Useful for certain statistics to ensure that a period such as January 31 (ending at 00z February 1) is included within January stats.
def convert_period_end_to_period_begin(data_array, temporal_res = 24):
    period_end_times = [pd.Timestamp(i) for i in data_array.period_end_time.values]
    period_begin_times = [i - pd.Timedelta(hours = temporal_res) for i in period_end_times]
   
    data_array_period_begin = data_array.rename({utils.period_end_time_dim_str: utils.period_begin_time_dim_str})
    data_array_period_begin.coords[utils.period_begin_time_dim_str] = period_begin_times 

    return data_array_period_begin

# Convert 'period_begin_time' dimension name and associated coordinates to 'period_end_time'
def convert_period_begin_to_period_end(data_array, temporal_res = 24):
    period_begin_times = [pd.Timestamp(i) for i in data_array.period_begin_time.values]
    period_end_times  = [i + pd.Timedelta(hours = temporal_res) for i in period_begin_times]

    data_array_period_end = data_array.rename({utils.period_begin_time_dim_str: utils.period_end_time_dim_str})
    data_array_period_end.coords[utils.period_end_time_dim_str] = period_end_times

    return data_array_period_end

# DATA DIRECTORY PATHS 
#################################################################################
# PSL Linux machines directories 
USER = "bbasarab"
home_dir = os.path.join("/home", USER) 
data_dir = os.path.join("/data", USER)
data_nc_dir = os.path.join(data_dir, "netcdf")
data_zarr_dir = os.path.join(data_dir, "zarr")
plot_output_dir = os.path.join(home_dir, "plots")
stats_output_dir = os.path.join(home_dir, "stats") 

# Template grids and regridders directories for xesmf interpolation
template_grids_dir = os.path.join(data_nc_dir, "TemplateGrids")
regridders_dir = os.path.join(data_nc_dir, "Regridders")

# Coarse/classic (0.25 degree resolution) UFS Replay on Google Cloud in zarr format
replay_coarse_gcs_path = "gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree/03h-freq/zarr/fv3.zarr"

# UNIT CONVERSIONS
#################################################################################

# Convert degree, minute, second strings with fmt <deg min' sec"> decimal degrees
def dms_strings_to_decdeg(lat_strings, long_strings, print_data = True):
    lat_degs = [float(istr.split()[0]) for istr in lat_strings]
    lat_mins = [float(istr.split()[1].strip("'")) for istr in lat_strings]
    lat_secs = [float(istr.split()[2].strip('"')) for istr in lat_strings]
    long_degs = [float(istr.split()[0]) for istr in long_strings]
    long_mins = [float(istr.split()[1].strip("'")) for istr in long_strings]
    long_secs = [float(istr.split()[2].strip('"')) for istr in long_strings]
    
    num_pts = len(lat_degs)

    lats_dec = [dms2decdeg(lat_degs[i], lat_mins[i], lat_secs[i]) for i in range(num_pts)]
    longs_dec = [dms2decdeg(long_degs[i], long_mins[i], long_secs[i]) for i in range(num_pts)]
    
    if print_data:
        print("Decimal latitudes:")
        for ilat in lats_dec:
            print(ilat) 
        print("Decimal longitudes:")
        for ilong in longs_dec:
            print(ilong)
 
# Degrees, minutes, seconds to decimal degrees
def dms_to_decdeg(degree, minute, second):
    if (degree < 0.) and (minute > 0.):
        minute = -minute
    if (degree < 0.) and (second > 0.):
        second = -second
    decdeg = degree + minute/60. + second/3600.
    return decdeg

# Convert Kelvin (native model temperature units) to Celsius
def degK_to_degC(degK):
    return degK - 273.15

def degC_to_degK(degC):
    return degK + 273.15

def degC_to_degF(degC):
    return degC * (9/5.) + 32

def degF_to_degC(degF):
    return (5/9.) * (degF - 32)

def feet_to_meters(feet):
    return feet * 0.3048

def meters_to_feet(meters):
    return meters * 3.28084

# Knots to meters per second
def knots_to_mps(knots):
    return knots * 0.5144

# Meters per second to knots
def mps_to_knots(mps):
    return mps * 1.94384

# Miles per hour to meters per second
def mph_to_mps(mph):
    return mph * 0.44704

# Meters per second to miles per hour
def mps_to_mph(mps):
   return mps * 2.23694

# LATITUDE AND LONGITUDE GRID CONVERSIONS
#################################################################################
def longitude_to_m180to180(lon_array):
    lon_array_new = np.copy(lon_array) 
    lon_array_new[lon_array_new > 180.0] -= 360.0
    return lon_array_new

def longitude_to_0to360(lon_array):
    lon_array_new = np.copy(lon_array)
    lon_array_new[lon_array_new < 0.0] += 360.0
    return lon_array_new

# TIME AND DATETIME CONVERSIONS 
#################################################################################
# Convert to and from unix timestamp; assumes datetime and unix_time are both UTC.
# NEW functions: explicitly put input and output datetimes in UTC
# This is necessary if the OS timezone is not UTC (e.g., PSL Linux servers are MT) 
def datetime2unix(dtime):
    return calendar.timegm(dtime.utctimetuple())

def unix2datetime(unix_time):
    return dt.datetime.fromtimestamp(unix_time, dt.UTC)

# NOTE: pd.Timestamp(dt64) also works and returns equivalent output (
def numpy_dt64_to_pandas_datetime(dt64):
    return pd.to_datetime(dt64)

def numpy_dt64_to_pandas_datetime_ymd(dt64):
    pd_ts = pd.Timestamp(dt64)
    pd_ts_ymd = pd.Timestamp(year = pd_ts.year, month = pd_ts.month, day = pd_ts.day)
    return pd_ts_ymd

def numpy_dt64_to_string(dt64, dt_format = "%Y%m%d.%H"):
    pandas_dt = pd.to_datetime(dt64)
    dt_str = pandas_dt.strftime(dt_format)  
    return dt_str

# Convert a timedelta object to minutes
def convert_timedelta_to_minutes(tdelta):
    duration_minutes = tdelta.days * 1440 + tdelta.seconds/60 + tdelta.microseconds/10**6/60
    return duration_minutes 

# OLD datetime <--> unix seconds functions: Don't work if your platform's timezone is not UTC (e.g., PSL Linux servers are MT)
#def datetime2unix(dtime, ret_int = True):
#    unix_time = time_module.mktime(dtime.timetuple())
#    if ret_int:
#        return int(unix_time)
#    return unix_time

#def unix2datetime(unix_time):
#    return dt.datetime.utcfromtimestamp(unix_time)

# INTERPOLATION 
#################################################################################
# Map intuitive strings representing interpolation type to the flag representing that
# type used by the cdo package.
def set_cdo_interpolation_type(args_flag):
    match args_flag:
        case "bilinear": # Bilinear interpolation
            return "remapbil"
        case "linear": # Bilinear interpolation
            return "remapbil"
        case "conservative": # First-order conservative interpolation
            return "remapcon"
        case "conservative2": # Second-order conservative interpolation
            return "remapcon2"
        case "nearest": # Nearest-neighbor interpolation
            return "remapnn"
        case _:
            print(f"Unrecognized interpolation type {args_flag}; will perform bilinear interpolation")
            return "remapbil"

def read_ds_template_for_xesmf_interp(ds_name):
    print(f"Reading template dataset for {ds_name}")

    ds_template_fpath = os.path.join(template_grids_dir, f"{ds_name}Grid.nc")
    if not(os.path.exists(ds_template_fpath)):
        print(f"Error: Template file path {ds_template_fpath} does not exist")
        sys.exit(1)
   
    ds = xr.open_dataset(ds_template_fpath)
    ds.attrs["ds_name"] = ds_name
    
    return ds

def add_bounds(ds, long_lat_lon_key_names = False):
    if long_lat_lon_key_names:
        key_list = ["latitude", "longitude"]
    else:
        key_list = ["lat", "lon"]

    ds = ds.cf.add_bounds(key_list)
    for key in key_list:
      corners = cfxr.bounds_to_vertices(bounds = ds[f"{key}_bounds"], bounds_dim = "bounds", order = None)
      ds = ds.assign_coords({f"{key}_b": corners})
      ds = ds.drop_vars(f"{key}_bounds")

    # In Tim Smith's code but doesn't work here (because we don't have the 'x_vertices' and
    # 'y_vertices' dimensions, so this step isn't necessary).
    # ds = ds.rename({"x_vertices": "x_b", "y_vertices": "y_b"})

    return ds

def run_xesmf_interp(in_ds_template, # Dataset on input grid 
                     out_ds_template, # Dataset on output grid
                     input_ds, # The actual data to regrid 
                     interp_method = "conservative",
                     is_curvilinear = True,
                     read_regridder = False,
                     long_lat_lon_key_names = False):

    # Add bounds to facilitate regridding from an input curvilinear grid
    if is_curvilinear:
        in_ds_template = utils.add_bounds(in_ds_template, long_lat_lon_key_names = long_lat_lon_key_names)

    regridder_fpath = os.path.join(regridders_dir,
                                   f"Regridder.{in_ds_template.ds_name}_to_{out_ds_template.ds_name}.{interp_method}.nc")

    if read_regridder: # Retrieve regridder from previously-written file
        print(f"Reading {interp_method} regridder {regridder_fpath}")
        if not(os.path.exists(regridder_fpath)):
            print(f"Error: Regridder file path {regridder_fpath} does not exist")
            sys.exit(1) 

        regridder = xesmf.Regridder(ds_in = in_ds_template,
                                    ds_out = out_ds_template,
                                    method = interp_method,
                                    weights = regridder_fpath)
    else: # Build regridder and write it to netCDF
        print(f"Building {interp_method} regridder")
        regridder = xesmf.Regridder(ds_in = in_ds_template,
                                    ds_out = out_ds_template,
                                    method = interp_method)

        print(f"Writing regridder to {regridder_fpath}")
        regridder.to_netcdf(regridder_fpath)
   
    # Run regridding 
    return regridder(input_ds, keep_attrs = True)

# XARRAY/NUMPY DATA ARRAY MANAGEMENT
#################################################################################
# TODO: Understand why the instances for which this function is necessary
# are dask arrays in the first place, and whether there's a more elegant way to
# handle them (can't call basic methods like .quantile(), .item() due to the structure of dask arrays)
def convert_from_dask_array(dask_array):
    da = xr.DataArray(dask_array.values, dims = dask_array.dims, coords = dask_array.coords,
                      attrs = dask_array.attrs)
    da.name = "accum_precip"

    return da

# Common renaming of latitude/longitude dimensions to the lat/lon names that I use
def rename_dims(data_array):
    return data_array.rename({
                             "latitude": "lat",
                             "longitude": "lon",
                             }) 

def print_basic_da_stats(da):
    print(f"Min: {da.min().item():0.2f}")  
    print(f"Max: {da.max().item():0.2f}")  
    print(f"Mean: {da.mean().item():0.2f}")  
    print(f"Median: {da.median().item():0.2f}")  
    print(f"95th pctl: {da.quantile(0.95).item():0.2f}")  
    print(f"99th pctl: {da.quantile(0.99).item():0.2f}")  
    print(f"99.9th pctl: {da.quantile(0.999).item():0.2f}")  

# COMMAND MANAGEMENT
#################################################################################
def run_cmd(cmd, log, testing = False):
    log.info(f"Executing: {cmd}")
    ret = 0
    if not testing:
        ret = os.system(cmd)
        log.info(f"Exit status: {ret}")
    return ret

def run_cmd_print(cmd, testing = False):
    print(f"Executing: {cmd}")
    ret = 0
    if not testing:
        ret = os.system(cmd)
        print(f"Exit status: {ret}")

def default_sigpipe():
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
 
# Communicate command output using subprocess
def run_subprocess_cmd(cmd_str):
    #log.info(cmd_str)
    #cmd_out = subprocess.Popen(cmd_str, stdout=subprocess.PIPE, shell = True).communicate()[0].strip()
    #p = subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE)
    p = subprocess.Popen(cmd_str, shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, preexec_fn=default_sigpipe)
    (cmd_out, cmd_err) = p.communicate()
    return (cmd_out.strip(), cmd_err)

def run_cmd_no_err_out(cmd_str):
    cmd_out = subprocess.Popen(cmd_str, stdout=subprocess.PIPE, shell = True).communicate()[0].strip()
    return cmd_out 

def remove_whitespace(string):
    return "".join(string.split())
