# A set of classes to process and interpolate various QPF/E datasets to a specified destination grid.
# Classes include:
    # ReplayDataProcessor: process Replay precip data to calculate accumulated precipitation amounts 
    # ImergDataProcesor: process IMERG precip data to calculate accumulated precipitation amounts; interpolate to destination grid
    # ERA5DataProcessor: process ERA5 precip data to calculate accumulated precipitation amounts; interpolate to destination grid
    # AorcDataProcessor: process AORC precip data to calculate accumulated precipitation amounts; interpolate to destination grid
    # CONUS404DataProcessor: process CONUS404 precip data to calculate accumulated precipitation amounts; interpolate to destination grid
    # NestedReplayDataProcessor: process Nested Replay precip data to calculate accumulated precipitation amounts; interpolate to destination grid 
    # StageIVDataProcessor: process StageIV precip data to calculate accumulated precipitation amounts; interpolate to destination grid

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime as dt
import dateutil
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import planetary_computer
import pygrib
import pystac_client
import sys
import utilities as utils
import xarray as xr

def add_attributes_to_data_array(data_array, short_name = None, long_name = None, units = None, interval_hours = None):
    if (short_name is not None):
        data_array.attrs["short_name"] = short_name 

    if (long_name is not None):
        data_array.attrs["long_name"] = long_name

    if (units is not None):
        data_array.attrs["units"] = units

    if (interval_hours is not None): 
        data_array.attrs["interval_hours"] = interval_hours 

# Calculate 24-hour precipitation from hourly precipitation
def calculate_24hr_accum_precip(input_data, output_temporal_res, native_temporal_res):
    time_step = int(output_temporal_res/native_temporal_res)
    roller = input_data.rolling({utils.period_end_time_dim_str: time_step})
    precip24 = roller.sum()[(time_step - 1)::time_step,:,:]
    precip24.name = utils.accum_precip_var_name
    add_attributes_to_data_array(precip24,
                                 short_name = f"24-hour precipitation",
                                 long_name = f"Precipitation accumulated over the prior 24 hour(s)",
                                 units = "mm",
                                 interval_hours = 24)

    return precip24

def check_model_valid_dt_format(dt_str, resolution = 3, check_resolution = True):
    if (resolution < 24):
        dt_fmt = "%Y%m%d.%H"
    else:
        dt_fmt = "%Y%m%d"

    try:
        valid_dt = dt.datetime.strptime(dt_str, dt_fmt)
    except ValueError:
        print(f"Error: Input datetime string {dt_str} does not match format {dt_fmt}, the required date/time format for {resolution}-hourly data")
        sys.exit(1)

    if (check_resolution) and (valid_dt.hour % resolution != 0):
        print(f"Error: Dataset is {resolution}-hourly; must provide a datetime at {resolution}-hourly intervals")
        sys.exit(1)

    return valid_dt

# Construct daily datetime list corresponding to cadence of daily netCDF files.
# This function is useful for translating from start_dt and end_dt, which define a time period
# based on period-ending dates, to corresponding daily dates for a series of netCDF files, with
# each date representing the day over which accumulated precip in the files is valid.
def construct_daily_datetime_list(start_dt, end_dt):
    current_daily_dt = dt.datetime(start_dt.year, start_dt.month, start_dt.day)
    end_daily_dt = dt.datetime(end_dt.year, end_dt.month, end_dt.day)

    valid_daily_dt_list = [current_daily_dt]
    while (current_daily_dt != end_daily_dt):
        current_daily_dt += dt.timedelta(days = 1)
        valid_daily_dt_list.append(current_daily_dt)

    return valid_daily_dt_list

# Return a new version of the short name attribute of a data array
# formatted to replace all spaces and dashes with underscores 
def format_short_name(data_array, make_lower = False):
    formatted_short_name = data_array.short_name.replace(' ', '_').replace('-', '_')
    if make_lower:
        return formatted_short_name.lower()
    return formatted_short_name

# Return the native temporal resolution of various datasets in hours
def get_dataset_native_temporal_res(data_name):
    match data_name:
        case "AORC":
            return 1
        case "CONUS404":
            return 1
        case "ERA5":
            return 3 # For in-house PSL ERA5 data
        case "HRRR":
            return 1
        case "IMERG":
            return 0.5
        case "NestedReplay":
            return 1
        case "Replay":
            return 3
        case _:
            return 1

# Read all CONUS404 variables from Azure as an xarray Dataset
def read_conus404_dataset_from_azure():
    print(f"Reading CONUS404 data from Azure into xarray Dataset")
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                        modifier = planetary_computer.sign_inplace)

    c = catalog.get_collection("conus404")
    asset = c.assets["zarr-abfs"]

    conus404_ds = xr.open_zarr(asset.href,
                               storage_options = asset.extra_fields["xarray:storage_options"],
                               **asset.extra_fields["xarray:open_kwargs"],
                               )

    return conus404_ds

# Spatially interpolate an xarray DataArray using xarray's interp_like method.
# Interpolate <input_data_array> to the grid of <destination_data_array>.
def spatially_interpolate_using_interp_like(input_data_array, destination_data_array,
                                            interp_method = "linear",
                                            correct_small_negative_values = True):
    data_array_destination_grid = input_data_array.interp_like(destination_data_array, method = interp_method)

    # Very small-magnitude negative values are introduced by this interpolation. Set values just below zero to zero.
    # Don't change larger-magnitude negative values, as these could indicate a legitimate problem.
    # Obviously, this correction is only applicable to variables that should not go below zero (like precipitation).
    if correct_small_negative_values:
        keep_condition = (data_array_destination_grid <= -0.001) | (data_array_destination_grid >= 0.0)
        data_array_destination_grid = data_array_destination_grid.where(keep_condition, 0.0) 

    print(f"Data array shape, interpolated to destination grid: {data_array_destination_grid.shape}")
    print(f"Destination data array shape (whose grid we're interpolating to): {destination_data_array.shape}")

    return data_array_destination_grid

# Read ARI grid
def read_ari_grid(ari, verif_grid, temporal_res = 24):
    output_grid_string = set_grid_name_for_file_names(verif_grid)
    ari_dir = os.path.join(utils.data_nc_dir, f"ARIs.{output_grid_string}")
    ari_fname = f"ARI.{output_grid_string}.{ari:04d}_year.{temporal_res:03d}_hour_precipitation.nc"
    ari_fpath = os.path.join(ari_dir, ari_fname)

    print(f"Reading {ari_fpath}")
    if not (os.path.exists(ari_fpath)):
        print(f"Error: ARI file {ari_fpath} does not exist")
        sys.exit(1)

    return xr.open_dataset(ari_fpath).precip

# Create string denoting the grid the dataset is on. It will
# always have the form {Dataset_name}{Grid}, where the first letter
# (and ONLY the first letter) of {Dataset_name} and {Grid} is capitalized. 
def set_grid_name_for_file_names(output_grid_name):
    return f"{output_grid_name.title()}Grid"

# Write an xarray DataArray to netCDF, with separate files according to the desired output file cadence. 
# The current choices that will work for file_cadence are:
    # 'interval_hours': write one file for each accumulation interval, e.g., 3-hourly data
    # 'day': write one file per day
    # Anything else: write all data to a single file
def write_data_array_to_netcdf(data_array, output_var_name, dir_name, fname_prefix, timestamp_format,
                               temporal_res = 3, file_cadence = "day", testing = False): 
        # If 24-hour precip timestamps are not valid at hour=00z, add a string to the directory and
        # file names denoting the hour at which 24-hour accumulated data starts (e.g., if 24-hour periods are from 12z-12z)
        hour_span_timestamp = ""
        first_timestamp = pd.Timestamp(data_array.period_end_time.values[0])
        if (temporal_res == 24) and (first_timestamp.hour != 0):
            hour_span_timestamp = f".{first_timestamp.hour:02d}z-{first_timestamp.hour:02d}z"
            dir_name += hour_span_timestamp

        if (not os.path.exists(dir_name)):
            print(f"Creating directory {dir_name}")
            if not(testing):
                os.mkdir(dir_name)

        if (file_cadence == "interval_hours"):
            for valid_dt in data_array["period_end_time"].values:
                # Configure timestamp to be used in output file name
                file_timestamp = pd.Timestamp(valid_dt)

                # Construct file path
                fname = f"{fname_prefix}.{file_timestamp.strftime(timestamp_format)}{hour_span_timestamp}.nc"
                fpath = os.path.join(dir_name, fname)

                # Write data to netCDF
                data_to_write = data_array.sel(period_end_time = valid_dt)
                print(f"Writing netCDF file {fpath}")
                if not(testing):
                    data_to_write.period_end_time.encoding["units"] = utils.seconds_since_unix_epoch_str 
                    data_to_write.to_netcdf(fpath, encoding = {output_var_name: {"dtype": "float32"}}) 
        elif (file_cadence == "day"):
            if (temporal_res == "native"):
                step = 24 * 2 # Assume half-hourly data
            else:
                step = int(24/temporal_res)

            start_day_index = 0
            end_day_index = step
            while (end_day_index <= data_array[utils.period_end_time_dim_str].shape[0]):
                # Configure timestamp to be used in output file name
                # Four 24-hourly data, adjust the timestamp used in the actual file names to be the day over which the data are valid.
                # Using the period_end_time coordinate data directly will result in files being timestamped with the NEXT day (confusing).
                # For example, 24-hour precip data valid over 20240101 ENDS on 20240102 (at 00:00:00). 
                file_timestamp = pd.Timestamp(data_array[utils.period_end_time_dim_str].values[start_day_index])
                if (temporal_res == 24):
                    file_timestamp -= pd.Timedelta(days = 1)

                fname = f"{fname_prefix}.{file_timestamp:%Y%m%d}{hour_span_timestamp}.nc"
                fpath = os.path.join(dir_name, fname)

                # Write data to netCDF
                data_to_write = data_array[start_day_index:end_day_index, :, :]
                print(f"Writing netCDF file {fpath}")
                if not(testing):
                    data_to_write.period_end_time.encoding["units"] = utils.seconds_since_unix_epoch_str 
                    data_to_write.to_netcdf(fpath, encoding = {output_var_name: {"dtype": "float32"}}) 

                start_day_index += step 
                end_day_index += step 
        else:
            # Configure timestamp to be used in output file name
            # Four 24-hourly data, adjust the timestamp used in the actual file names to be the day over which the data are valid.
            # Using the period_end_time coordinate data directly will result in files being timestamped with the NEXT day (confusing).
            # For example, 24-hour precip data valid over 20240101 ENDS on 20240102 (at 00:00:00). 
            start_file_timestamp = pd.Timestamp(data_array[utils.period_end_time_dim_str].values[0])
            end_file_timestamp = pd.Timestamp(data_array[utils.period_end_time_dim_str].values[-1])
            if (temporal_res == 24):
                start_file_timestamp -= pd.Timedelta(days = 1)
                end_file_timestamp -= pd.Timedelta(days = 1)
        
            file_timestamp = f"{start_file_timestamp.strftime(timestamp_format)}-{end_file_timestamp.strftime(timestamp_format)}"

            fname = f"{fname_prefix}.{file_timestamp}{hour_span_timestamp}.nc"
            fpath = os.path.join(dir_name, fname)

            # Write data to netCDF
            data_to_write = data_array
            print(f"Writing netCDF file {fpath}")
            if not(testing):
                data_to_write.period_end_time.encoding["units"] = utils.seconds_since_unix_epoch_str 
                data_to_write.to_netcdf(fpath, encoding = {output_var_name: {"dtype": "float32"}})

class ReplayDataProcessor(object):
    def __init__(self,
                 start_dt_str,
                 end_dt_str,
                 input_variable_list = ["prateb_ave"],
                 model_name = "Replay",
                 input_format = "zarr", # zarr or nc
                 temporal_res = 3,
                 DEST_GRID_FLAG = False, # Set to True to spatially interpolate to destination grid 
                 dest_temporal_res = 24, # Temporal resolution of data we want to spatially interpolate to destination grid
                 dest_grid_name = "AORC",
                 interp_method = "linear",
                 obs_name = None, 
                 lat_lon = None,
                 region = "Global",
                 user_dir = "bbasarab"):
 
        # Process input variable list
        if (len(input_variable_list) == 0):
            print("Error: Must provide at least one input variable")
            sys.exit(1)
        self.input_variable_list = input_variable_list 

        # Check date formats
        self.obs_name = obs_name
        self.input_format = input_format 
        self.temporal_res = int(temporal_res)
        self.DEST_GRID_FLAG = DEST_GRID_FLAG
        start_dt_str, end_dt_str = self._convert_obs_dates_to_model_dates(start_dt_str, end_dt_str)
        self.start_dt = check_model_valid_dt_format(start_dt_str, resolution = self.temporal_res)
        self.end_dt = check_model_valid_dt_format(end_dt_str, resolution = self.temporal_res)
        self.start_dt_str = start_dt_str
        self.end_dt_str = end_dt_str
        self.time_period_str = f"{self.start_dt:%Y%m%d.%H}-{self.end_dt:%Y%m%d.%H}"

        self.time_dim_name = utils.time_dim_str 
        self.period_begin_time_dim_str = utils.period_begin_time_dim_str 
        self.period_end_time_dim_str = utils.period_end_time_dim_str 

        # Check input latitudes/longitudes
        self.flag_for_time_series = self._check_lat_lon_for_timeseries(lat_lon)

        # Configure other parameters
        self.model_name = model_name 
        self.region = region
        self.user_dir = user_dir
        self.home_dir = os.path.join("/home", self.user_dir)
        self.data_dir = os.path.join("/data", self.user_dir) 
        self.plot_output_dir = os.path.join(self.home_dir, "plots") 
        self.netcdf_output_dir = os.path.join(self.data_dir, "netcdf")        

        # Read Replay dataset
        if (self.input_format == "zarr"):
            self._read_replay_dataset_from_gcp()
        else:
            self._read_replay_dataset_from_netcdf()

        # Data processing
        self._index_variables_by_datetime()
        self._calculate_sum_over_time_period()

        # If DEST_GRID_FLAG = True, define the parameters of the separate grid to which we'll interpolate. 
        if self.DEST_GRID_FLAG:
            self.dest_grid_name = dest_grid_name
            self.dest_grid_native_temporal_res = get_dataset_native_temporal_res(self.dest_grid_name) # Native temporal resolution of destination grid
            self.dest_temporal_res = dest_temporal_res
            self.interp_method = interp_method
            self._spatially_interpolate_replay_to_dest_grid()

    ##### GETTER METHODS (ReplayDataProcessor) #####
    def get_replay_variable_list(self):
        return self.variables_data_dict.keys()

    def get_replay_name(self):
        return self.model_name

    def get_replay_data_array(self, var_name, summed = False):
        if summed:
            return self.summed_time_period_dict[var_name]
        return self.variables_data_dict[var_name]

    def get_replay_var_long_name(self, var_name, summed = False):
        if summed:
            return self.summed_time_period_dict[var_name].long_name
        return self.variables_data_dict[var_name].long_name

    def get_replay_var_short_name(self, var_name, summed = False):
        if summed:
            return self.summed_time_period_dict[var_name].short_name
        return self.variables_data_dict[var_name].short_name

    def get_replay_var_units(self, var_name, summed = False):
        if summed:
            return self.summed_time_perioed_dict[var_name].units
        return self.variables_data_dict[var_name].units

    def get_precip_data(self, temporal_res = 3, spatial_res = "native", load = False):
        if (spatial_res == "dest_grid") and not(self.DEST_GRID_FLAG):
            print(f"No data at {spatial_res} spatial resolution")
            return

        if (temporal_res == self.temporal_res) and (spatial_res == "native"):
            data_array = self.variables_data_dict[utils.accum_precip_var_name]
        else:
            data_array = self._calculate_replay_accum_precip_amount(temporal_res = temporal_res,
                                                                    spatial_res = spatial_res)

        if load:
            print(f"Loading precip data for {self.model_name}")
            data_array.load()

        return data_array 
    
    def how_to_get_precip_data(self):
        print("**** HOW TO GET REPLAY PRECIP DATA:")
        print("Call get_precip_data(temporal_res = 3, spatial_res = 'native', load = False)")
        print("The argument temporal_res is an integer representing the desired temporal resolution in hours")
    
    ##### PRIVATE METHODS (ReplayDataProcessor) #####
    # If we're going to plot a time series, ensure we have the data we need to do so.
    def _check_lat_lon_for_timeseries(self, lat_lon):
        self.plot_lat = None
        self.plot_lon = None
        self.lat_lon_tuple_str = None

        if lat_lon is None: 
            return False 

        # Ensure lat/lon entries can be interpreted numerically 
        try:
            plot_lat = float(lat_lon[0])
            plot_lon = float(lat_lon[1])
        except:
            print(f"Warning: Latitude {lat_lon[0]} or longitude {lat_lon[1]} are invalid; not plotting timeseries at point")
            return False 
 
        # Check for unphysical latitudes; replay latitudes go from (90, -90)
        if np.abs(plot_lat) >= 90:
            print(f"Warning: Invalid latitude {plot_lat}; not plotting timeseries at point")
            return False

        # Check for unphysical longitudes
        if (np.abs(plot_lon) >= 360) or (plot_lon < -180):
            print(f"Error: Invalid longitude {plot_lon}; not plotting timeseries at point")
            return False 

        # Replay longitudes go from [0, 360), so convert negative (but valid) input longitudes accordingly
        if (0 > plot_lon >= -180):
            plot_lon += 360

        self.plot_lat = plot_lat
        self.plot_lon = plot_lon
        self.lat_lon_tuple_str = f"({self.plot_lat:0.2f}, {self.plot_lon:0.2f})" 
        return True

    # Create correct corresponding start and end dates, based on 
    # start and end dates of corresponding observational data.
    # For example, Replay data are 3-hourly; IMERG data are half-hourly.
    def _convert_obs_dates_to_model_dates(self, start_dt_str, end_dt_str):
        match self.obs_name:
            case "IMERG":
                obs_dt_format = "%Y%m%d.%H%M"
                obs_dt_example_string = "YYYYmmdd.HHMM"
                obs_temporal_res_hours = 0.5     
            case _:
                return start_dt_str, end_dt_str

        try:
            if (len(start_dt_str) != len(obs_dt_example_string)) or \
               (len(end_dt_str) != len(obs_dt_example_string)):
                sys.exit(1)
            obs_start_dt = dt.datetime.strptime(start_dt_str, obs_dt_format) 
            obs_end_dt = dt.datetime.strptime(end_dt_str, obs_dt_format)
        except:
            print(f"Error: Input datetime strings {start_dt_str} and {end_dt_str} are not formatted properly for conversion to model start/end datetimes; must be {obs_dt_example_string}")
            sys.exit(1) 
        
        model_start_dt = dt.datetime(obs_start_dt.year,
                                     obs_start_dt.month,
                                     obs_start_dt.day,
                                     obs_start_dt.hour) + \
                                     dt.timedelta(hours = self.temporal_res)
        model_end_dt = obs_end_dt + dt.timedelta(hours = obs_temporal_res_hours)

        model_start_dt_str = model_start_dt.strftime("%Y%m%d.%H")
        model_end_dt_str = model_end_dt.strftime("%Y%m%d.%H")
        return model_start_dt_str, model_end_dt_str

    def _index_variables_by_datetime(self):
        self.variables_data_dict = {}
        for var in self.input_variable_list:
            # Get data for current variable
            current_variable = self.atmos_dataset[var].rename({"grid_xt": "lon", "grid_yt": "lat"})
            if ("long_name" in current_variable.attrs):
                current_variable.attrs["short_name"] = current_variable.long_name
      
            # Save data indexed across the desired date/time range 
            self.variables_data_dict[var] = current_variable.loc[f"{self.start_dt:%Y-%m-%d %H:%M:%S}":f"{self.end_dt:%Y-%m-%d %H:%M:%S}"]

            # Calculate accumulated precipitation from prateb_ave, if prateb_ave is
            # available in self.input_variable_list
            self._calculate_replay_native_accum_precip(var)

    def _read_replay_dataset_from_gcp(self):
        # Read in the entire (!!) Replay dataset (from 1994-2023)
        print(f"Reading {self.model_name} data from GCP into xarray Dataset")
        self.atmos_dataset = xr.open_zarr(utils.replay_coarse_gcs_path, storage_options={"token": "anon"})
    
    # Useful to read Nested Replay global domain data         
    def _read_replay_dataset_from_netcdf(self):
        print(f"Reading {self.model_name} data from netCDF")
        input_dir = f"{self.model_name}.NativeGrid.{self.temporal_res:02d}_hour_precipitation"
        input_dir = os.path.join(utils.data_nc_dir, input_dir)
        nc_string = os.path.join(input_dir, "*.nc")

        self.atmos_dataset = xr.open_mfdataset(nc_string)
    
    # Calculate accumulated precip at the native temporal resolution of the Replay
    # That is accumulated precip valid over the time steps of the model 
    def _calculate_replay_native_accum_precip(self, var_name):
        is_precip = False
        match var_name:
            case "prateb_ave":
                print(f"Calculating accumulated {self.model_name} precipitation from {var_name}")
                accum_precip_data_array = self.variables_data_dict[var_name] * 3600 * self.temporal_res
                is_precip = True
            case "accum_precip": # For daily files from Nested Replay global domain
                print(f"Using {self.model_name} {var_name} directly for precipitation")
                accum_precip_data_array = self.variables_data_dict[var_name]
                is_precip = True

        # Process precipitation data
        if is_precip:
            accum_precip_data_array = accum_precip_data_array.rename({utils.time_dim_str: utils.period_end_time_dim_str}) 
          
            add_attributes_to_data_array(accum_precip_data_array,
                                         short_name = f"{self.temporal_res:02d}-hour precipitation",
                                         long_name = f"Precipitation accumulated over the prior {self.temporal_res} hour(s)",
                                         units = "mm",
                                         interval_hours = self.temporal_res) 
            accum_precip_data_array.name = utils.accum_precip_var_name
            self.replay_native_accum_precip = accum_precip_data_array 
            self.variables_data_dict[utils.accum_precip_var_name] = self.replay_native_accum_precip 

    # Sum totals at the native replay temporal resolution to derive totals over longer time periods
    # (3, 6, 12, 24 hours...).
    def _calculate_replay_accum_precip_amount(self, temporal_res = 3, spatial_res = "native"):
        if (spatial_res == "native"):
            raw_data = self.replay_native_accum_precip
        else:
            if not(self.DEST_GRID_FLAG): 
                print(f"No data at {spatial_res} spatial resolution")
                return
            raw_data = self.precip_dest_grid

        time_step = int(temporal_res/self.temporal_res)
        roller = raw_data.rolling({self.period_end_time_dim_str: time_step})
        accum_precip_data_array = roller.sum()[(time_step - 1)::time_step,:,:]

        add_attributes_to_data_array(accum_precip_data_array,
                                     short_name = f"{temporal_res:02d}-hour precipitation",
                                     long_name = f"Precipitation accumulated over the prior {temporal_res} hour(s)",
                                     units = self.variables_data_dict[utils.accum_precip_var_name].units,
                                     interval_hours = temporal_res)

        return accum_precip_data_array
    
    def _spatially_interpolate_replay_to_dest_grid(self):
        if (not self.DEST_GRID_FLAG): 
            print("No destination grid specified; not spatially interpolating")
            return 
        print(f"Spatially interpolating {self.model_name} data to {self.dest_grid_name}")

        # Select Replay data at the native temporal resolution (3 hours) to interpolate.
        # Calculation of accmulated amounts at different temporal resolutions is handled by
        # _calculate_replay_accum_precip_amount (called within get_precip_data).
        replay_precip_to_interpolate = self.get_precip_data(spatial_res = "native", temporal_res = self.temporal_res).copy()
        
        # Change longitude coordinates of Replay to go from [-180, 180] to match AORC coordinates.
        if (self.dest_grid_name == "AORC"):
            replay_lons_m180to180 = utils.longitude_to_m180to180(replay_precip_to_interpolate["lon"].values)
            replay_precip_to_interpolate["lon"] = replay_lons_m180to180
            replay_precip_to_interpolate = replay_precip_to_interpolate.sortby("lon") 

        # Get destination data array by instantiating the appropriate class
        match self.dest_grid_name:
            case "AORC":
                start_dt_dest_grid = self.start_dt - dt.timedelta(hours = self.temporal_res - self.dest_grid_native_temporal_res)
                end_dt_dest_grid = self.end_dt
                dest_grid_processor = AorcDataProcessor(start_dt_dest_grid.strftime("%Y%m%d.%H"),
                                                        end_dt_dest_grid.strftime("%Y%m%d.%H"),
                                                        DEST_GRID_FLAG = False,
                                                        region = self.region)
                self.dest_data_array = dest_grid_processor.get_precip_data(spatial_res = "native", temporal_res = self.temporal_res) 
            case "Replay": # 'Interpolate' Nested Replay global domain data to its own grid, for proper naming of output files
                self.precip_dest_grid = replay_precip_to_interpolate.copy()
                return
            case "StageIV":
                print("Error: Interpolation from {self.model_name} grid to {self.dest_grid_name} grid not yet implemented")
                sys.exit(1)
            case _:
                print(f"Error: Interpolation from {self.model_name} grid to {self.dest_grid_name} grid not yet implemented")
                sys.exit(1)

        # Spatially interpolate Replay data to output grid 
        self.precip_dest_grid = spatially_interpolate_using_interp_like(replay_precip_to_interpolate,
                                                                        self.dest_data_array,
                                                                        interp_method = self.interp_method, 
                                                                        correct_small_negative_values = True)   

    # Calculate sum over time period: take sum along date/time dimension at all spatial points  
    def _calculate_sum_over_time_period(self):
        self.summed_time_period_dict = {}
        for var in self.get_replay_variable_list():
            data_array = self.get_replay_data_array(var)
            if (utils.period_end_time_dim_str in data_array.dims): 
                summed_data_array = data_array.sum(dim = utils.period_end_time_dim_str)
                num_time_intervals = data_array[utils.period_end_time_dim_str].shape[0]
            else: 
                summed_data_array = data_array.sum(dim = utils.time_dim_str)
                num_time_intervals = data_array[utils.time_dim_str].shape[0]

            # Set attributes properly
            time_period = num_time_intervals * self.temporal_res
            if (var == utils.accum_precip_var_name):
                short_name = f"{time_period:02d}-hour precipitation"
                long_name = f"Precipitation accumulated over the prior {time_period} hour(s)"
            else:
                short_name = self.get_replay_var_long_name(var)
                long_name = data_array.long_name    
            add_attributes_to_data_array(summed_data_array,
                                         short_name = short_name,
                                         long_name = long_name,
                                         units = data_array.units,
                                         interval_hours = time_period)

            self.summed_time_period_dict[var] = summed_data_array 

    ##### PUBLIC METHODS (ReplayDataProcessor) #####
    # Calculate mean of data summed over time period (useful for verifying mean total precip per day, etc.) 
    def calculate_mean_of_sum_over_time_period(self, var_name):
        return self.get_replay_data_array(var_name, summed = True).mean(dim = ["lon", "lat"])
  
    # Calculate mean of data summed over time period where data are non-zero 
    def calculate_mean_of_sum_over_time_period_nonzero(self, var_name):
        data_array = self.get_replay_data_array(var_name, summed = True)
        return data_array.where(data_array > 0.0).mean(dim = ["lon", "lat"])

    # Take time series at a single lat/long point
    def calculate_time_series_at_point(self, var_name):
        if (not self.flag_for_time_series):
            print(f"Insufficient info to extract time series at point (e.g., missing lat/long)")
            return

        data_array = self.get_replay_data_array(var_name)
        time_series_point = data_array.sel(lon = self.plot_lon, lat = self.plot_lat, method = "nearest")
        return time_series_point

    # Calculate sum at a single lat/long point 
    def calculate_sum_at_point(self, var_name):
        if (not self.flag_for_time_series):
            print(f"Insufficient info to calculate sum at point (e.g., missing lat/long)")
            return
        time_series_point = self.calculate_time_series_at_point(var_name)
        summed_time_period_point = time_series_point.sum()
        return summed_time_period_point 

    def plot_replay_data_time_series(self, var_name):
        if (not self.flag_for_time_series):
            print(f"Insufficient info to create time series plot (e.g., missing lat/long)")
            return
        print(f"Plotting timeseries of {self.model_name} {var_name} from {self.time_period_str} at {self.lat_lon_tuple_str}") 
        data_to_plot = self.calculate_time_series_at_point(var_name) 
        print(f"Number of data in time series: {data_to_plot.shape[0]}")
        
        long_name = self.get_replay_var_long_name(var_name)
        units = self.get_replay_var_units(var_name)
        title_string = f"{self.model_name}: {var_name} at {self.lat_lon_tuple_str} from {self.time_period_str}"
        fig_name = f"{self.model_name}.timeseries.{var_name}.lat{self.plot_lat:0.2f}.lon{self.plot_lon:0.2f}.{self.time_period_str}.png"
       
        # Set up the figure and plot data 
        data_to_plot.plot(figsize = (15, 10), linewidth = 3) 
        plt.grid(True, linewidth = 0.5)
        plt.tick_params(axis = "both", labelsize = 12)
        plt.title(title_string, fontsize = 15)
        plt.xlabel("Date/Time", fontsize = 15)
        plt.ylabel(f"{long_name} [{units}]", fontsize = 15)

        # Save figure
        fig_path = os.path.join(self.plot_output_dir, fig_name)
        print(f"Saving {fig_path}")
        plt.savefig(fig_path)

    def write_precip_data_to_netcdf(self, temporal_res = 3, spatial_res = "native", file_cadence = "day", testing = False):
        if (spatial_res == "dest_grid") and (not self.DEST_GRID_FLAG):
            print(f"No data at {spatial_res} spatial resolution to write to netCDF")
            return

        # Get data to write
        full_data_array = self.get_precip_data(temporal_res = temporal_res, spatial_res = spatial_res, load = True)

        output_var_name = f"precipitation_{temporal_res:02d}_hour"
        formatted_short_name = f"{temporal_res:02d}_hour_precipitation"
        full_data_array.name = output_var_name
        
        # Construct file directory
        if (spatial_res == "native"):
            output_grid_string = set_grid_name_for_file_names(spatial_res)
        elif (spatial_res == "dest_grid"):
            output_grid_string = set_grid_name_for_file_names(self.dest_grid_name)
        fname_prefix = f"{self.model_name}.{output_grid_string}.{formatted_short_name}"
        dir_name = os.path.join(self.netcdf_output_dir, fname_prefix)
        
        # Construct timestamp format
        if (temporal_res < 24):
            timestamp_format = "%Y%m%d.%H"
        else:
            timestamp_format = "%Y%m%d"
        
        write_data_array_to_netcdf(full_data_array,
                                   output_var_name,
                                   dir_name,
                                   fname_prefix,
                                   timestamp_format,
                                   temporal_res = temporal_res,
                                   file_cadence = file_cadence,
                                   testing = testing)

class ImergDataProcessor(object):
    def __init__(self,
                 start_dt_str,
                 end_dt_str,
                 obs_name = "IMERG",
                 native_data_dir = "/Projects/BIL/Extreme_Intercomp/IMERG/Final",
                 native_data_version_string = "V07B",
                 native_data_temporal_res_hours = 0.5,
                 DEST_GRID_FLAG = False, # Set to True to spatially interpolate to destination grid 
                 dest_temporal_res = 24, # Temporal resolution of data we want to spatially interpolate to destination grid
                 dest_grid_name = "Replay",
                 interp_method = "linear",
                 region = "Global",
                 user_dir = "bbasarab"):

        # Process start and end datetime strings into all the date/time info we'll need
        self.DEST_GRID_FLAG = DEST_GRID_FLAG
        self.native_data_temporal_res_hours = native_data_temporal_res_hours # Always in hours
        self.start_dt_str = start_dt_str
        self.end_dt_str = end_dt_str
        self.start_dt = self._check_imerg_valid_dt_format(self.start_dt_str)
        self.end_dt = self._check_imerg_valid_dt_format(self.end_dt_str)
        self.start_dt_period_end = self.start_dt + dt.timedelta(hours = self.native_data_temporal_res_hours)
        self.end_dt_period_end = self.end_dt + dt.timedelta(hours = self.native_data_temporal_res_hours) 

        # Set time dimension names
        self.time_dim_name = utils.time_dim_str 
        self.period_begin_time_dim_str = utils.period_begin_time_dim_str 
        self.period_end_time_dim_str = utils.period_end_time_dim_str
 
        # File reading/writing and plotting
        self.obs_name = obs_name
        self.native_data_dir = native_data_dir
        self.native_data_version_string = native_data_version_string
        self.region = region
        self.user_dir = user_dir
        self.home_dir = os.path.join("/home", self.user_dir)
        self.data_dir = os.path.join("/data", self.user_dir) 
        self.plot_output_dir = os.path.join(self.home_dir, "plots") 
        self.netcdf_output_dir = os.path.join(self.data_dir, "netcdf")        

        # Read native precipitation data that we will process 
        self._construct_valid_dt_list()
        self._construct_input_file_list()
        self._read_input_files()

        # Call data processing methods
        self._create_precip_data_array()
        self._calculate_all_native_accum_precip_amounts()

        # Set destination grid to interpolate to
        if self.DEST_GRID_FLAG:
            self.dest_grid_name = dest_grid_name
            self.dest_grid_native_temporal_res = get_dataset_native_temporal_res(self.dest_grid_name) # Native temporal resolution of destination grid
            self.dest_temporal_res = dest_temporal_res # Output temporal resolution that we want for output to netCDF, etc.
            self.interp_method = interp_method
            self._spatially_interpolate_imerg_to_dest_grid()

    ##### GETTER METHODS (ImergDataProcessor) #####
    def get_imerg_valid_dt_list(self):
        return self.valid_dt_list

    def get_imerg_input_file_list(self):
        return self.input_file_list

    def get_imerg_full_xr_dataset(self):
        return self.xr_dataset

    def get_precip_data(self, spatial_res = "native", temporal_res = "native", load = False):
        match spatial_res:
            case "dest_grid":
                if (not self.DEST_GRID_FLAG): 
                    print(f"No data at {spatial_res} spatial resolution")
                    return
                data_array = self._calculate_imerg_accum_precip_amount(time_period_hours = temporal_res, spatial_res = spatial_res) 
                self._add_imerg_data_array_attributes(data_array, temporal_res) 
            case "native":
                match temporal_res:
                    case "native":
                        data_array = self.native_precip_data
                    case _:
                        data_array = self._calculate_imerg_accum_precip_amount(time_period_hours = temporal_res)
                        self._add_imerg_data_array_attributes(data_array, temporal_res) 
            case _:
                print(f"No data at {spatial_res} spatial resolution; use 'native' or 'dest_grid'")
                return
            
        if load:
            if (spatial_res == "native"):
                print(f"Loading precip data for {self.obs_name}")
            else:
                print(f"Loading precip data for {self.obs_name} interpolated to {self.dest_grid_name} grid")
            data_array.load()
        return data_array

    def get_imerg_missing_value(self):
        return self.missing_value

    def get_dest_grid_name(self):
        if (not self.DEST_GRID_FLAG):
            print("No accompanying dest grid data")
            return 
        return self.dest_grid_name

    def get_obs_name(self):
        return self.obs_name
    
    def how_to_get_precip_data(self):
        print("**** HOW TO GET IMERG PRECIP DATA:")
        print("Call get_precip_data(spatial_res = 'native', temporal_res = 'native', load = False)")
        print("The argument spatial_res can be 'dest_grid' or 'native'")
        print("The argument temporal_res can be 'native' or an integer representing the desired temporal resolution in hours")
 
    ##### PRIVATE METHODS (ImergDataProcessor) #####
    def _check_imerg_valid_dt_format(self, dt_str, exit = True):
        try:
            date_string = dt_str.split(".")[0]
            time_string = dt_str.split(".")[1] 
            if (len(date_string) != 8) or (len(time_string) != 4): 
                print(f"Error: Invalid valid date/time input format {dt_str}; must be YYYYmmdd.HHMM")
                sys.exit(1)
            valid_dt = dt.datetime.strptime(dt_str, "%Y%m%d.%H%M") 
        except:
            print(f"Error: Invalid valid date/time input format {dt_str}; must be YYYYmmdd.HHMM")
            if exit:
                sys.exit(1)
            return

        return valid_dt

    def _construct_valid_dt_list(self):
        # Datetimes valid at beginning of each period
        current_dt = self.start_dt
        self.valid_dt_list = [current_dt]
        while (current_dt != self.end_dt):
            current_dt += dt.timedelta(hours = self.native_data_temporal_res_hours)
            self.valid_dt_list.append(current_dt)

        # Now construct datetime valid at end of each period
        current_dt = self.start_dt_period_end
        self.valid_dt_list_period_end = [current_dt]
        while (current_dt != self.end_dt_period_end):
            current_dt += dt.timedelta(hours = self.native_data_temporal_res_hours)
            self.valid_dt_list_period_end.append(current_dt) 

    def _construct_end_of_single_file_time(self, start_valid_dt):
        return start_valid_dt + dt.timedelta(seconds = 29 * 60 + 59)
             
    # Number of minutes into the day (so from 0000-1410) is needed
    # since it's part of the file name. 
    def _construct_minutes_into_day_string(self, valid_dt):
        valid_dt_beginning_of_day = dt.datetime(valid_dt.year, valid_dt.month, valid_dt.day, 0, 0)
        valid_seconds_into_day = utils.datetime2unix(valid_dt) - utils.datetime2unix(valid_dt_beginning_of_day)
        valid_minutes_into_day = int(valid_seconds_into_day/60.)
        valid_minutes_into_day_str = f"{valid_minutes_into_day:04d}"

        return valid_minutes_into_day_str

    def _construct_input_file_list(self):
        self.input_file_list = []
        for valid_dt in self.valid_dt_list:
            end_valid_dt = self._construct_end_of_single_file_time(valid_dt)
            valid_minutes_into_day_str = self._construct_minutes_into_day_string(valid_dt) 
            file_base_pattern = f"3B-HHR.MS.MRG.3IMERG.{valid_dt:%Y%m%d}-S{valid_dt:%H%M%S}-E{end_valid_dt:%H%M%S}.{valid_minutes_into_day_str}.{self.native_data_version_string}.HDF5"
            file_path = os.path.join(self.native_data_dir, f"{valid_dt:%Y}", file_base_pattern)

            # Proceed to next file if a file doesn't exist (but need to handle any data gaps correctly)
            if (not os.path.exists(file_path)):
                print(f"Warning: Input file path {file_path} does not exist")
                continue
            
            self.input_file_list.append(file_path)

    def _read_input_files(self):
        print(f"Reading {self.obs_name} data from netCDF files into xarray Dataset")
        self.xr_dataset = xr.open_mfdataset(self.input_file_list, group = "Grid")

    def _add_imerg_data_array_attributes(self, data_array, time_period_hours, units = "mm"):
        data_array.attrs["short_name"] = f"{time_period_hours:02d}-hour precipitation"
        data_array.attrs["long_name"] = f"Precipitation accumulated over the prior {time_period_hours} hour(s)"
        data_array.attrs["units"] = units
    
    # Imerg data are half-hourly averages of precip rate (mm/hr). To derive an hourly amount (mm),
    # calculate (1/2 hour) * (rate in period :00-:30) + (1/2 hour) * (rate in period :30-:60)
    # which, mathematically, is the same as averaging the two half-hourly rates.
    # Note this code assumes the first valid time is from :00-:30. Then we simply average
    # data at indices [0, 1]; [2, 3]; [4, 5];.... This assumption needs to be generalized.
    def _calculate_hourly_precip_amount(self): 
        # To get the total precip valid at the end of the current hour:
            # 1) Take the sum over the two values valid during that hour. This is accomplished via a combination of creating a rolling object and then calling .sum() on it.
            # 2) Start at first index (works because times are valid period ending), skip by 2 to get summed values valid at top of hour and over that previous full hour.
            # 3) Multiply by 0.5 (i.e., divide by 2) to take the average of all the summed values. 
        #roller = self.native_precip_data.rolling(period_end_time = 2) # This syntax also works
        roller = self.native_precip_data.rolling({self.period_end_time_dim_str: 2})
        precip_hourly_amount = 0.5 * roller.sum()[1::2,:,:] 

        return precip_hourly_amount

    # Sum hourly totals to derive totals over longer time periods
    # (3, 6, 12, 24 hours...). Again, some code here needs to be generalized.
    def _calculate_imerg_accum_precip_amount(self, time_period_hours = 3, spatial_res = "native"):
        if (spatial_res == "dest_grid"):
            time_step = int(time_period_hours/self.dest_grid_native_temporal_res)
            raw_data = self.precip_dest_grid
        else:
            time_step = int(time_period_hours/1) # Use IMERG hourly data to calculate other accumulation amounts
            raw_data = self.precip_hourly
        roller = raw_data.rolling({self.period_end_time_dim_str: time_step})
        accum_precip_data_array = roller.sum()[(time_step - 1)::time_step,:,:]

        return accum_precip_data_array

    def _calculate_all_native_accum_precip_amounts(self):
        print(f"Calculating {self.obs_name} accumulated precipitation amounts on {self.obs_name} grid")

        self.precip_hourly = self._calculate_hourly_precip_amount()
        self._add_imerg_data_array_attributes(self.precip_hourly, 1)
        
        self.precip_3hourly = self._calculate_imerg_accum_precip_amount(time_period_hours = 3)
        self._add_imerg_data_array_attributes(self.precip_3hourly, 3)

        self.precip_6hourly = self._calculate_imerg_accum_precip_amount(time_period_hours = 6)
        self._add_imerg_data_array_attributes(self.precip_6hourly, 6)

        self.precip_12hourly = self._calculate_imerg_accum_precip_amount(time_period_hours = 12)
        self._add_imerg_data_array_attributes(self.precip_12hourly, 12)

        self.precip_24hourly = self._calculate_imerg_accum_precip_amount(time_period_hours = 24)
        self._add_imerg_data_array_attributes(self.precip_24hourly, 24)

    # Create isolated DataArray containing precip data for easier manipulation
    # Also ensure that the time dimension is valid at the END of each period
    def _create_precip_data_array(self):
        self.native_precip_data = self.xr_dataset.precipitation.transpose("time", "lat", "lon"). \
                                                                rename({"time": self.period_end_time_dim_str})
        
        # Create arrays representing the time coordinates: arrays of cftimes, numpy datetime64 objects, and pandas Timestamps.
        # Then replace the cftimes coordinates in the native precip data array with numpy datetime64 coordinates. 
        # Also, importantly, convert time coordinates to period-ending.
        # NOTE: uncomfortable with the .indexes method to convert from cftimes to numpy datetime64. 
        # Is there a more elegant/more understandable way to do this?
 
        # Cftimes arrays
        self.cftimes = self.native_precip_data[self.period_end_time_dim_str].values
        self.cftimes_period_end = self.cftimes + dt.timedelta(hours = self.native_data_temporal_res_hours)
        
        # Numpy datetime64 arrays
        self.dt64 = self.native_precip_data.indexes[self.period_end_time_dim_str].to_datetimeindex()
        self.dt64_period_end = [i + pd.Timedelta(hours = self.native_data_temporal_res_hours) for i in self.dt64]

        # Pandas Timestamp arrays
        self.pd_timestamps = [pd.Timestamp(i) for i in self.dt64]
        self.pd_timestamps_period_end = [pd.Timestamp(i) for i in self.dt64_period_end]

        # Finally replace the cftimes coordinates for the time dimension with numpy datetim64 coordinates.
        self.native_precip_data.coords[self.period_end_time_dim_str] = self.dt64_period_end

        # Add and clean up DataArray attributes 
        self.missing_value = float(self.xr_dataset.precipitation.CodeMissingValue) 
        self.native_precip_data.attrs["missing_value"] = self.missing_value
        self.native_precip_data = self.native_precip_data.where(self.native_precip_data != self.missing_value)

        add_attributes_to_data_array(self.native_precip_data,
                                     short_name = "Precipitation rate",
                                     long_name = "Average precipitation rate valid over prior time period")

        del self.native_precip_data.attrs["LongName"] # Bunch of weird spaces and returns in the original long name, so get rid of it
        del self.native_precip_data.attrs["CodeMissingValue"] # Redundant with missing value attribute we add above
        del self.native_precip_data.attrs["Units"] # Redundant attribute

    # Spatially interopolate IMERG precipitation data to a different grid, typically
    # a model grid to use it in verification of that model. Use xarray's interp_like method.
    # To be able to use interp_like, we need to ensure that:
        # 1) The *names* ('lat', 'lon', etc.) of the source and destination grid dimensions are the same.
        # 2) The *coordinates* (i.e., actual data) of the source and destination grid dimensions are the same.
            # For example, if the source grid represents lons as [-180, 180] and the destination grid represents
            # lons as [0, 360], the longitude coordinates of one of the grids must be changed such that both are
            # either [-180, 180] or [0, 360].
        # 3) Both grids must be re-ordered/re-indexed such that the order of the dimensions is the same in each. 
            # NOTE: On 20241120, this step was found to be unnecessary for interp_like to work properly. Notice
            # that below, longitudes are re-ordered but latitudes are not. Also refer to demo_interp_conus404.py.
            # When interp_like runs, it will reorder the coordinates of the source grid to match the order of the
            # destination grid. However, I am keeping this re-ordering step in order to make the code more
            # transparent, intuitive, and understandable. 
    # Step 1) is done in _index_variables_by_datetime in the ReplayDataProcessor class, where the Replay dimensions are renamed.
    # Step 2) is done below, in _spatially_interpolate_imerg_to_dest_grid (calculation of imerg_lons_0to360 and assignment to "lon" coordinates).
    # Step 3) is done below, in _spatially_interpolate_imerg_to_dest_grid (call to sortby method).
    def _spatially_interpolate_imerg_to_dest_grid(self):
        if (not self.DEST_GRID_FLAG): 
            print("No dest grid data; not spatially interpolating to dest grid grid")
            return 
        print(f"Spatially interpolating {self.obs_name} data to {self.dest_grid_name} data grid")

        imerg_precip_to_interpolate = self.get_precip_data(spatial_res = "native",
                                                           temporal_res = self.dest_grid_native_temporal_res).copy()

        # Change longitude coordinates of IMERG to go from [0, 360] to match Replay coordinates.
        if (self.dest_grid_name == "Replay"):
            imerg_lons_0to360 = utils.longitude_to_0to360(imerg_precip_to_interpolate["lon"].values)
            imerg_precip_to_interpolate["lon"] = imerg_lons_0to360 
            imerg_precip_to_interpolate = imerg_precip_to_interpolate.sortby("lon")
 
        # Get destination data array by instantiating the appropriate class
        match self.dest_grid_name:
            case "AORC":
                start_dt_dest_grid = self.start_dt + dt.timedelta(hours = 1) # IMERG is period beginning, starts at .0000
                end_dt_dest_grid = self.end_dt + dt.timedelta(minutes = 30) # IMERG is period beginning, ends at .XX30
                dest_grid_processor = AorcDataProcessor(start_dt_dest_grid.strftime("%Y%m%d.%H"),
                                                        end_dt_dest_grid.strftime("%Y%m%d.%H"),
                                                        DEST_GRID_FLAG = False,
                                                        region = self.region)
            case "Replay":
                start_dt_dest_grid = self.start_dt + dt.timedelta(hours = 3)
                end_dt_dest_grid = self.end_dt + dt.timedelta(minutes = 30) 
                dest_grid_processor = ReplayDataProcessor(start_dt_dest_grid.strftime("%Y%m%d.%H"),
                                                          end_dt_dest_grid.strftime("%Y%m%d.%H"), 
                                                          DEST_GRID_FLAG = False,
                                                          region = self.region)
            case "StageIV":
                print("Error: Interpolation to StageIV grid not yet implemented")
                return 
            case _:
                print(f"Error: Interpolation to {self.dest_grid_name} grid not yet implemented")
                return 

        # Spatially interpolate IMERG data to the dest grid
        self.dest_data_array = dest_grid_processor.get_precip_data(spatial_res = "native", temporal_res = self.dest_grid_native_temporal_res)
        self.precip_dest_grid = spatially_interpolate_using_interp_like(imerg_precip_to_interpolate,
                                                                        self.dest_data_array, 
                                                                        interp_method = self.interp_method, 
                                                                        correct_small_negative_values = True)   

    ##### PUBLIC METHODS (ImergDataProcessor) #####
    def write_precip_data_to_netcdf(self, temporal_res = "native", spatial_res = "native", file_cadence = "day", testing = False):
        if (spatial_res == "dest_grid") and (not self.DEST_GRID_FLAG):
            print(f"No data at {spatial_res} spatial resoluation to write to netCDF")
            return 

        # Get data to write
        full_data_array = self.get_precip_data(temporal_res = temporal_res, spatial_res = spatial_res, load = True)

        if (temporal_res == "native"):
            output_var_name = "precipitation"
            formatted_short_name = format_short_name(full_data_array)
        else:
            output_var_name = f"precipitation_{temporal_res:02d}_hour"
            formatted_short_name = f"{temporal_res:02d}_hour_precipitation"
        full_data_array.name = output_var_name
        
        if (spatial_res == "native"):
            output_grid_string = set_grid_name_for_file_names("native")
        elif (spatial_res == "dest_grid"):
            output_grid_string = set_grid_name_for_file_names(self.dest_grid_name)
        
        # Construct file directory
        fname_prefix = f"{self.obs_name}.{output_grid_string}.{formatted_short_name}"
        dir_name = os.path.join(self.netcdf_output_dir, fname_prefix)
        
        # Construct timestamp format
        if (temporal_res == "native"):
            timestamp_format = "%Y%m%d.%H%M"
        elif (temporal_res < 24):
            timestamp_format = "%Y%m%d.%H"
        else:
            timestamp_format = "%Y%m%d"
        
        write_data_array_to_netcdf(full_data_array,
                                   output_var_name,
                                   dir_name,
                                   fname_prefix,
                                   timestamp_format,
                                   temporal_res = temporal_res,
                                   file_cadence = file_cadence,
                                   testing = testing)

class ERA5DataProcessor(object):
    def __init__(self,
                 start_dt_str,
                 end_dt_str,
                 obs_name = "ERA5",
                 native_data_dir = "/Projects/era5/monolevel/",
                 temporal_res = 3, # ERA5 is hourly, but the data availabale at PSL are 3-hourly
                 input_variable_list = ["prate"],
                 DEST_GRID_FLAG = False, # Set to True to spatially interpolate to destination grid 
                 dest_temporal_res = 24, # Temporal resolution of data we want to spatially interpolate to destination grid
                 dest_grid_name = "Replay",
                 interp_method = "linear",
                 region = "Global",
                 user_dir = "bbasarab"):

        # Process start and end datetime strings into all the date/time info we'll need
        self.DEST_GRID_FLAG = DEST_GRID_FLAG
        self.temporal_res = temporal_res
        self.input_variable_list = input_variable_list
        self.start_dt_str = start_dt_str
        self.end_dt_str = end_dt_str
        self.start_dt = check_model_valid_dt_format(self.start_dt_str, resolution = 3)
        self.end_dt = check_model_valid_dt_format(self.end_dt_str, resolution = 3)
        self.start_dt_period_end = self.start_dt + dt.timedelta(hours = self.temporal_res)
        self.end_dt_period_end = self.end_dt + dt.timedelta(hours = self.temporal_res)
        self.start_dt_str_period_end = self.start_dt_period_end.strftime("%Y%m%d.%H")
        self.end_dt_str_period_end = self.end_dt_period_end.strftime("%Y%m%d.%H")
        self.first_year_dt = dt.datetime(self.start_dt.year, 1, 1)
        self.final_year_dt = dt.datetime(self.end_dt.year, 1, 1)

        # Set time dimension names
        self.time_dim_name = utils.time_dim_str 
        self.period_begin_time_dim_str = utils.period_begin_time_dim_str 
        self.period_end_time_dim_str = utils.period_end_time_dim_str
 
        # File reading/writing and plotting
        self.obs_name = obs_name
        self.native_data_dir = native_data_dir
        self.region = region
        self.user_dir = user_dir
        self.home_dir = os.path.join("/home", self.user_dir)
        self.data_dir = os.path.join("/data", self.user_dir) 
        self.plot_output_dir = os.path.join(self.home_dir, "plots") 
        self.netcdf_output_dir = os.path.join(self.data_dir, "netcdf")        

        # Read native precipitation data that we will process 
        self._construct_valid_dt_lists()
        self._construct_input_file_lists()
        self._read_input_files()

        # Set an accompanying dest grid to interpolate to
        if self.DEST_GRID_FLAG:
            self.dest_grid_name = dest_grid_name
            self.dest_temporal_res = dest_temporal_res
            self.interp_method = interp_method
            self._spatially_interpolate_era5_to_dest_grid()

    ##### GETTER METHODS (ERA5DataProcessor) #####
    def get_precip_data(self, spatial_res = "native", temporal_res = "native", load = False):
        match spatial_res:
            case "dest_grid":
                if (not self.DEST_GRID_FLAG): 
                    print(f"No data at {spatial_res} spatial resolution")
                    return
                data_array = self._calculate_era5_accum_precip_amount(time_period_hours = temporal_res, spatial_res = spatial_res) 
            case "native":
                match temporal_res:
                    case "native":
                        data_array = self.era5_da_dict[utils.accum_precip_var_name]
                    case _:
                        data_array = self._calculate_era5_accum_precip_amount(time_period_hours = temporal_res)
            case _:
                print(f"No data at {spatial_res} spatial resolution; use 'native' or 'dest_grid'")
                return

        if load:
            print(f"Loading precip data for {self.obs_name}")
            data_array.load()

        return data_array
 
    def get_dest_grid_name(self):
        if (not self.DEST_GRID_FLAG):
            print("No accompanying dest_grid data")
            return 
        return self.dest_grid_name

    def get_obs_name(self):
        return self.obs_name
    
    def how_to_get_precip_data(self):
        print("**** HOW TO GET ERA5 PRECIP DATA:")
        print("Call get_precip_data(spatial_res = 'native', temporal_res = 'native', load = False)")
        print("The argument spatial_res can be 'dest_grid' or 'native'")
        print("The argument temporal_res can be 'native' or an integer representing the desired temporal resolution in hours")
 
    ##### PRIVATE METHODS (ERA5DataProcessor) #####
    def _construct_valid_dt_lists(self):
        # Datetimes valid at beginning of each period
        current_dt = self.start_dt
        self.valid_dt_list = [current_dt]
        while (current_dt != self.end_dt):
            current_dt += dt.timedelta(hours = self.temporal_res)
            self.valid_dt_list.append(current_dt)

        # Now construct datetime valid at end of each period
        current_dt = self.start_dt_period_end
        self.valid_dt_list_period_end = [current_dt]
        while (current_dt != self.end_dt_period_end):
            current_dt += dt.timedelta(hours = self.temporal_res)
            self.valid_dt_list_period_end.append(current_dt)

        # Now construct list of valid years (PSL ERA5 files separated by year)
        self.yearly_dt_list = list(dateutil.rrule.rrule(dateutil.rrule.YEARLY, dtstart = self.first_year_dt, until = self.final_year_dt))

    def _construct_input_file_lists(self):
        self.file_list_for_each_var = {}
        for var in self.input_variable_list:
            input_file_list = []
            for valid_year in self.yearly_dt_list:
                    file_base_pattern = f"{var}.{valid_year:%Y}.nc"
                    file_path = os.path.join(self.native_data_dir, file_base_pattern)

                    # Proceed to next file if a file doesn't exist (but need to handle any data gaps correctly)
                    if (not os.path.exists(file_path)):
                        print(f"Warning: Input file path {file_path} does not exist")
                        continue
                    
                    input_file_list.append(file_path)
            self.file_list_for_each_var[var] = input_file_list

    def _read_input_files(self):
        print(f"Reading {self.obs_name} data from netCDF files into xarray Dataset")
        self.era5_da_dict = {}
        for var in self.input_variable_list:
            xr_dataset = xr.open_mfdataset(self.file_list_for_each_var[var])
            current_variable = xr_dataset[var]

            # Save data indexed across the desired date/time range 
            self.era5_da_dict[var] = current_variable.loc[f"{self.start_dt:%Y-%m-%d %H:%M:%S}":f"{self.end_dt:%Y-%m-%d %H:%M:%S}"]

            # Calculate accumulated precipitation from prate, if prate is
            # available in self.input_variable_list
            self._calculate_era5_native_accum_precip_amount(var)

    # Calculate accumulated precip at the native temporal resolution ERA5 
    # That is accumulated precip valid over the time steps of ERA5 
    def _calculate_era5_native_accum_precip_amount(self, var_name):
        match var_name:
            case "prate":
                print(f"Calculating accumulated {self.obs_name} precipitation from {var_name}")

                # Multiply 3-hourly precip rate (in kg/m^2/s) by three hours (in seconds) to derive 3-hourly amount
                accum_precip_data_array = self.era5_da_dict[var_name] * 3600 * self.temporal_res

                # Change the name of the time dimension from "time" to "period_end_time"
                accum_precip_data_array = accum_precip_data_array.rename({utils.time_dim_str: utils.period_end_time_dim_str}) 
    
                # Replace the time coordinates with period end times (we want to intepret precip amounts as valid period
                # ending for consistency with the Replay grid.
                accum_precip_data_array.coords[self.period_end_time_dim_str] = self.valid_dt_list_period_end 
               
                # Add attributes to the accumulated precip data array
                add_attributes_to_data_array(accum_precip_data_array,
                                             short_name = f"{self.temporal_res:02d}-hour precipitation",
                                             long_name = f"Precipitation accumulated over the prior {self.temporal_res} hour(s)",
                                             units = "mm",
                                             interval_hours = self.temporal_res) 
                accum_precip_data_array.name = utils.accum_precip_var_name

                # Add accumulated precip data array to the dictionary of input variables
                self.era5_da_dict[utils.accum_precip_var_name] = accum_precip_data_array 
            case _:
                pass

    # Sum totals at the ERA5 temporal resolution to derive totals over longer time periods
    # (3, 6, 12, 24 hours...).
    def _calculate_era5_accum_precip_amount(self, time_period_hours = 3, spatial_res = "native"):
        time_step = int(time_period_hours/self.temporal_res)

        if (spatial_res == "dest_grid"):
            raw_data = self.precip_dest_grid
        else:
            raw_data = self.era5_da_dict[utils.accum_precip_var_name]
        roller = raw_data.rolling({self.period_end_time_dim_str: time_step})
        accum_precip_data_array = roller.sum()[(time_step - 1)::time_step,:,:]

        add_attributes_to_data_array(accum_precip_data_array,
                                     short_name = f"{time_period_hours:02d}-hour precipitation",
                                     long_name = f"Precipitation accumulated over the prior {time_period_hours} hour(s)",
                                     units = self.era5_da_dict[utils.accum_precip_var_name].units,
                                     interval_hours = time_period_hours)

        return accum_precip_data_array

    def _spatially_interpolate_era5_to_dest_grid(self):
        if (not self.DEST_GRID_FLAG): 
            print("No dest grid data; not spatially interpolating to {self.dest_grid_name} grid")
            return 
        print(f"Spatially interpolating {self.obs_name} data to {self.dest_grid_name} grid")

        era5_precip_to_interpolate = self.get_precip_data(spatial_res = "native",
                                                          temporal_res = self.temporal_res).copy()
        
        # Change longitude coordinates of ERA5 to go from [-180, 180] (rather than [0, 360]) to match AORC coordinates 
        if (self.dest_grid_name == "AORC"):
            era5_lons_m180to180 = utils.longitude_to_m180to180(era5_precip_to_interpolate["lon"].values)
            era5_precip_to_interpolate["lon"] = era5_lons_m180to180
            era5_precip_to_interpolate = era5_precip_to_interpolate.sortby("lon") 

        # Get destination data array by instantiating the appropriate class
        match self.dest_grid_name:
            case "AORC":
                start_dt_dest_grid = self.start_dt + dt.timedelta(hours = 1) # ERA5 is period beginning, starts at .0000 
                end_dt_dest_grid = self.end_dt + dt.timedelta(hours = 3) # ERA5 is period beginnning, ends at at .2100 
                dest_grid_processor = AorcDataProcessor(start_dt_dest_grid.strftime("%Y%m%d.%H"),
                                                        end_dt_dest_grid.strftime("%Y%m%d.%H"),
                                                        DEST_GRID_FLAG = False,
                                                        region = self.region)
            case "Replay":
                start_dt_dest_grid = self.start_dt + dt.timedelta(hours = 3)
                end_dt_dest_grid = self.end_dt + dt.timedelta(hours = 3)
                dest_grid_processor = ReplayDataProcessor(start_dt_dest_grid.strftime("%Y%m%d.%H"),
                                                          end_dt_dest_grid.strftime("%Y%m%d.%H"),
                                                          DEST_GRID_FLAG = False,
                                                          region = self.region)
            case "StageIV":
                print("Error: Interpolation to StageIV grid not yet implemented")
                return 
            case _:
                print(f"Error: Interpolation to {self.dest_grid_name} grid not yet implemented")
                return 

        self.dest_data_array = dest_grid_processor.get_precip_data(spatial_res = "native", temporal_res = self.temporal_res) 
        self.precip_dest_grid = spatially_interpolate_using_interp_like(era5_precip_to_interpolate,
                                                                        self.dest_data_array,
                                                                        interp_method = self.interp_method, 
                                                                        correct_small_negative_values = True)   
    
    ##### PUBLIC METHODS (ERA5DataProcessor) #####
    def write_precip_data_to_netcdf(self, temporal_res = "native", spatial_res = "native", file_cadence = "day", testing = False):
        if (spatial_res == "dest_grid") and (not self.DEST_GRID_FLAG):
            print(f"No data at {spatial_res} spatial resoluation to write to netCDF")
            return 

        # Get data to write
        full_data_array = self.get_precip_data(temporal_res = temporal_res, spatial_res = spatial_res, load = True)

        if (temporal_res == "native"):
            output_var_name = "precipitation"
            formatted_short_name = format_short_name(full_data_array)
        else:
            output_var_name = f"precipitation_{temporal_res:02d}_hour"
            formatted_short_name = f"{temporal_res:02d}_hour_precipitation"
        full_data_array.name = output_var_name
        
        if (spatial_res == "native"):
            output_grid_string = set_grid_name_for_file_names("native") 
        elif (spatial_res == "dest_grid"):
            output_grid_string = set_grid_name_for_file_names(self.dest_grid_name)
 
        # Construct file directory
        fname_prefix = f"{self.obs_name}.{output_grid_string}.{formatted_short_name}"
        dir_name = os.path.join(self.netcdf_output_dir, fname_prefix)
        
        # Construct timestamp format
        if (temporal_res == "native"):
            timestamp_format = "%Y%m%d.%H%M"
        elif (temporal_res < 24):
            timestamp_format = "%Y%m%d.%H"
        else:
            timestamp_format = "%Y%m%d"
        
        write_data_array_to_netcdf(full_data_array,
                                   output_var_name,
                                   dir_name,
                                   fname_prefix,
                                   timestamp_format,
                                   temporal_res = temporal_res,
                                   file_cadence = file_cadence,
                                   testing = testing)

class AorcDataProcessor(object):
    def __init__(self,
                 start_dt_str,
                 end_dt_str,
                 obs_name = "AORC",
                 #native_data_dir = "/Projects/BIL/Extreme_Intercomp/AORC", # /Projects/BIL area with time offset
                 native_data_dir = "/Projects/AORC_CONUS_4km",
                 obs_temporal_res = 1, 
                 DEST_GRID_FLAG = False, # Set to True to spatially interpolate to destination grid 
                 dest_temporal_res = 24, # Temporal resolution of data we want to spatially interpolate to destination grid
                 dest_grid_name = "Replay",
                 interp_method = "linear",
                 region = "Global",
                 user_dir = "bbasarab"):

        # Process start and end datetime strings into all the date/time info we'll need
        self.DEST_GRID_FLAG = DEST_GRID_FLAG
        self.obs_temporal_res = obs_temporal_res
        self.start_dt_str = start_dt_str
        self.end_dt_str = end_dt_str
        self.start_dt = check_model_valid_dt_format(self.start_dt_str, resolution = 1)
        self.end_dt = check_model_valid_dt_format(self.end_dt_str, resolution = 1)
        self.first_day_dt = dt.datetime(self.start_dt.year, self.start_dt.month, self.start_dt.day)
        self.final_day_dt = dt.datetime(self.end_dt.year, self.end_dt.month, self.end_dt.day)

        # Set time dimension names
        self.time_dim_name = utils.time_dim_str 
        self.period_begin_time_dim_str = utils.period_begin_time_dim_str 
        self.period_end_time_dim_str = utils.period_end_time_dim_str
 
        # File reading/writing and plotting
        self.obs_name = obs_name
        self.native_data_dir = native_data_dir
        self.region = region
        self.user_dir = user_dir
        self.home_dir = os.path.join("/home", self.user_dir)
        self.data_dir = os.path.join("/data", self.user_dir) 
        self.plot_output_dir = os.path.join(self.home_dir, "plots") 
        self.netcdf_output_dir = os.path.join(self.data_dir, "netcdf")        

        # Read native precipitation data that we will process 
        self._construct_valid_dt_lists()
        self._construct_input_file_list()
        self._read_input_files()

        # Extract hourly accumulated precipitation amounts (native AORC data are hourly) 
        self._calculate_aorc_native_accum_precip_amount()

        # Set an accompanying dest grid to interpolate to
        if self.DEST_GRID_FLAG:
            self.dest_grid_name = dest_grid_name
            self.dest_grid_native_temporal_res = get_dataset_native_temporal_res(self.dest_grid_name) # Native temporal resolution of destination grid
            self.dest_temporal_res = dest_temporal_res
            self.interp_method = interp_method
            self._spatially_interpolate_aorc_to_dest_grid()

    ##### GETTER METHODS (AorcDataProcessor) #####
    # FIXME: Here and other get_precip_data methods: Don't try to return a precip array if spatial_res = "dest_grid"
    # but temporal_res is finer than the dest_grid resolution (since this isn't possible)
    def get_precip_data(self, spatial_res = "native", temporal_res = "native", load = False):
        match spatial_res:
            case "dest_grid":
                if (not self.DEST_GRID_FLAG): 
                    print(f"No data at {spatial_res} spatial resolution")
                    return
                data_array = self._calculate_aorc_accum_precip_amount(time_period_hours = temporal_res, spatial_res = spatial_res) 
            case "native":
                match temporal_res:
                    case "native":
                        data_array = self.aorc_native_accum_precip 
                    case _:
                        data_array = self._calculate_aorc_accum_precip_amount(time_period_hours = temporal_res)
            case _:
                print(f"No data at {spatial_res} spatial resolution; use 'native' or 'dest_grid'")
                return

        if load:
            print(f"Loading precip data for {self.obs_name}")
            data_array.load()

        return data_array
 
    def get_dest_grid_name(self):
        if (not self.DEST_GRID_FLAG):
            print("No accompanying dest grid data")
            return 
        return self.dest_grid_name

    def get_obs_name(self):
        return self.obs_name
 
    def how_to_get_precip_data(self):
        print("**** HOW TO GET AORC PRECIP DATA:")
        print("Call get_precip_data(spatial_res = 'native', temporal_res = 'native', load = False)")
        print("The argument spatial_res can be 'dest_grid' or 'native'")
        print("The argument temporal_res can be 'native' or an integer representing the desired temporal resolution in hours")

    ##### PRIVATE METHODS (AorcDataProcessor) #####
    def _construct_valid_dt_lists(self):
        # Construct datetimes valid at end of each period (this is how AORC data are presented)
        current_dt = self.start_dt
        self.valid_dt_list = [current_dt]
        while (current_dt != self.end_dt):
            current_dt += dt.timedelta(hours = self.obs_temporal_res)
            self.valid_dt_list.append(current_dt)

        # Now construct list of valid days (PSL AORC files separated by day)
        self.daily_dt_list = list(dateutil.rrule.rrule(dateutil.rrule.DAILY, dtstart = self.first_day_dt, until = self.final_day_dt))

    def _construct_input_file_list(self):
        self.input_file_list = []
        for valid_dt in self.daily_dt_list:
            #file_base_pattern = f"{self.obs_name}.{valid_dt:%Y%m%d}.precip.nc" # /Projects/BIL area with time offset
            file_base_pattern = f"prate.aorc.{valid_dt:%Y%m%d}.nc"
            file_path = os.path.join(self.native_data_dir, f"{valid_dt:%Y}", file_base_pattern)

            # Proceed to next file if a file doesn't exist (but need to handle any data gaps correctly)
            if (not os.path.exists(file_path)):
                print(f"Warning: Input file path {file_path} does not exist")
                continue

            self.input_file_list.append(file_path)
            
    def _read_input_files(self):
        print(f"Reading {self.obs_name} data from netCDF files into xarray Dataset")
        self.xr_dataset = xr.open_mfdataset(self.input_file_list)

    # Calculate accumulated precip at the native temporal resolution of AORC (i.e., hourly)
    # AORC presents average precip rate in mm/hr over the prior hour, so just used these values directly to represent hourly accumulated precipitation 
    def _calculate_aorc_native_accum_precip_amount(self):
        print(f"Calculating accumulated {self.obs_name} precipitation from prate")

        # Use precip rate data directly, because it's average hourly precip rate (mm/hr) over the prior hour 
        # accum_precip_data_array = self.xr_dataset["precrate"].loc[f"{self.start_dt:%Y-%m-%d %H:%M:%S}":f"{self.end_dt:%Y-%m-%d %H:%M:%S}"] # /Projects/BIL area with time offset
        accum_precip_data_array = self.xr_dataset["prate"].loc[f"{self.start_dt:%Y-%m-%d %H:%M:%S}":f"{self.end_dt:%Y-%m-%d %H:%M:%S}"]

        # Replace missing values with NaNs
        self.missing_value = float(self.xr_dataset.attrs["missing"]) 
        accum_precip_data_array = accum_precip_data_array.where(accum_precip_data_array != self.missing_value)

        # Change the name of the time dimension from "time" to "period_end_time"
        accum_precip_data_array = accum_precip_data_array.rename({utils.time_dim_str: utils.period_end_time_dim_str,
                                                                  "latitude": "lat",
                                                                  "longitude": "lon",  
                                                                  }) 
       
        # Add attributes to the accumulated precip data array
        add_attributes_to_data_array(accum_precip_data_array,
                                     short_name = f"{self.obs_temporal_res:02d}-hour precipitation",
                                     long_name = f"Precipitation accumulated over the prior {self.obs_temporal_res} hour(s)",
                                     units = "mm",
                                     interval_hours = self.obs_temporal_res) 
        accum_precip_data_array.name = utils.accum_precip_var_name

        self.aorc_native_accum_precip = accum_precip_data_array 

    # Sum totals at the AORC temporal resolution to derive totals over longer time periods
    # (3, 6, 12, 24 hours...).
    def _calculate_aorc_accum_precip_amount(self, time_period_hours = 3, spatial_res = "native"):
        if (spatial_res == "dest_grid"):
            time_step = int(time_period_hours/self.dest_temporal_res)
            raw_data = self.precip_dest_grid
        else:
            time_step = int(time_period_hours/self.obs_temporal_res)
            raw_data = self.aorc_native_accum_precip
        roller = raw_data.rolling({self.period_end_time_dim_str: time_step})
        accum_precip_data_array = roller.sum()[(time_step - 1)::time_step,:,:]

        add_attributes_to_data_array(accum_precip_data_array,
                                     short_name = f"{time_period_hours:02d}-hour precipitation",
                                     long_name = f"Precipitation accumulated over the prior {time_period_hours} hour(s)",
                                     units = self.aorc_native_accum_precip.units,
                                     interval_hours = time_period_hours)

        return accum_precip_data_array

    def _spatially_interpolate_aorc_to_dest_grid(self):
        if (not self.DEST_GRID_FLAG): 
            print(f"No dest grid data; not spatially interpolating to {self.dest_grid} grid")
            return 
        print(f"Spatially interpolating {self.obs_name} data to {self.dest_grid} grid")

        aorc_precip_to_interpolate = self.get_precip_data(spatial_res = "native",
                                                          temporal_res = self.dest_grid_native_temporal_res).copy()

        # Change longitude coordinates of AORC to be a subset of [0, 360] to match Replay coordinates.
        if (self.dest_grid_name == "Replay"):
            aorc_lons_0to360 = utils.longitude_to_0to360(aorc_precip_to_interpolate["lon"].values)
            aorc_precip_to_interpolate["lon"] = aorc_lons_0to360 
            aorc_precip_to_interpolate = aorc_precip_to_interpolate.sortby("lon") 

        # Get destination data array by instantiating the appropriate class
        match self.dest_grid_name:
            case "Replay":
                start_dt_dest_grid = self.start_dt + dt.timedelta(hours = self.dest_grid_native_temporal_res - self.temporal_res)
                end_dt_dest_grid = self.end_dt
                dest_grid_processor = ReplayDataProcessor(start_dt_dest_grid.strftime("%Y%m%d.%H"),
                                                          end_dt_dest_grid.strftime("%Y%m%d.%H"),
                                                          DEST_GRID_FLAG = False,
                                                          region = self.region)
            case "StageIV":
                print("Error: Interpolation to StageIV grid not yet implemented")
                return 
            case _:
                print(f"Error: Interpolation to {self.dest_grid_name} grid not yet implemented")
                return

        self.dest_data_array = dest_grid_processor.get_precip_data(spatial_res = "native", temporal_res = self.dest_grid_native_temporal_res)
        self.precip_dest_grid = spatially_interpolate_using_interp_like(aorc_precip_to_interpolate,
                                                                        self.dest_data_array,
                                                                        interp_method = self.interp_method, 
                                                                        correct_small_negative_values = True)   
    
    ##### PUBLIC METHODS (AorcDataProcessor) #####
    def write_precip_data_to_netcdf(self, temporal_res = "native", spatial_res = "native", file_cadence = "day", testing = False):
        if (spatial_res == "dest_grid") and (not self.DEST_GRID_FLAG):
            print(f"No data at {spatial_res} spatial resoluation to write to netCDF")
            return 

        # Get data to write
        full_data_array = self.get_precip_data(temporal_res = temporal_res, spatial_res = spatial_res, load = True)

        if (temporal_res == "native"):
            output_var_name = "precipitation"
            formatted_short_name = format_short_name(full_data_array)
        else:
            output_var_name = f"precipitation_{temporal_res:02d}_hour"
            formatted_short_name = f"{temporal_res:02d}_hour_precipitation"
        full_data_array.name = output_var_name
        
        if (spatial_res == "native"):
            output_grid_string = set_grid_name_for_file_names(spatial_res)
        elif (spatial_res == "dest_grid"):
            output_grid_string = set_grid_name_for_file_names(self.dest_grid_name)

        # Construct file directory
        fname_prefix = f"{self.obs_name}.{output_grid_string}.{formatted_short_name}"
        dir_name = os.path.join(self.netcdf_output_dir, fname_prefix)
        
        # Construct timestamp format
        if (temporal_res == "native"):
            timestamp_format = "%Y%m%d.%H%M"
        elif (temporal_res < 24):
            timestamp_format = "%Y%m%d.%H"
        else:
            timestamp_format = "%Y%m%d"
        
        write_data_array_to_netcdf(full_data_array,
                                   output_var_name,
                                   dir_name,
                                   fname_prefix,
                                   timestamp_format,
                                   temporal_res = temporal_res,
                                   file_cadence = file_cadence,
                                   testing = testing)

# NOTE: This class currently reads and loads CONUS404 data from Microsoft Planetary Computer.
# CONUS404 precip data only (hourly precip) is also on the PSL Linux servers: /Projects/HydroMet/CONUS404/
class CONUS404DataProcessor(object):
    def __init__(self,
                 start_dt_str,
                 end_dt_str,
                 data_name = "CONUS404",
                 input_variable_list = ["PREC_ACC_NC"],
                 read_from_azure = True, # Set to True to read data from Azure; otherwise read native grid netCDF files
                 native_temporal_res = 1, 
                 DEST_GRID_FLAG = False, # Set to True to spatially interpolate to destination grid 
                 dest_temporal_res = 24, # Temporal resolution of data we want to spatially interpolate to destination grid
                 dest_grid_name = "Replay",
                 interp_method = "linear",
                 region = "CONUS",
                 user_dir = "bbasarab"):
 
        # Process input variable list
        if (len(input_variable_list) == 0):
            print("Error: Must provide at least one input variable")
            sys.exit(1)
        self.input_variable_list = input_variable_list

        # Process start and end datetime strings into all the date/time info we'll need
        self.DEST_GRID_FLAG = DEST_GRID_FLAG
        self.native_temporal_res = native_temporal_res
        self.start_dt_str = start_dt_str
        self.end_dt_str = end_dt_str
        self.start_dt = check_model_valid_dt_format(self.start_dt_str, resolution = 1)
        self.end_dt = check_model_valid_dt_format(self.end_dt_str, resolution = 1)
        self.first_day_dt = dt.datetime(self.start_dt.year, self.start_dt.month, self.start_dt.day)
        self.final_day_dt = dt.datetime(self.end_dt.year, self.end_dt.month, self.end_dt.day)

        # Set time dimension names
        self.time_dim_name = utils.time_dim_str 
        self.period_begin_time_dim_str = utils.period_begin_time_dim_str 
        self.period_end_time_dim_str = utils.period_end_time_dim_str
 
        # File reading/writing and plotting
        self.data_name = data_name
        self.region = region
        self.user_dir = user_dir
        self.home_dir = os.path.join("/home", self.user_dir)
        self.data_dir = os.path.join("/data", self.user_dir) 
        self.plot_output_dir = os.path.join(self.home_dir, "plots") 
        self.netcdf_dir = os.path.join(self.data_dir, "netcdf")
        self.dest_temporal_res = dest_temporal_res

        # If DEST_GRID_FLAG = True, define the parameters of the separate grid to which we'll
        # interpolate. Otherwise, read native CONUS404 dataset from Azure or from existing native grid
        # netCDF files. We only need to read data from Azure if creating native grid files, since files on a 
        # destination grid will be interpolated from previously-created native grid netCDF files using cdo.
        if self.DEST_GRID_FLAG:
            self.dest_grid_name = dest_grid_name
            self.interp_method = interp_method
        else:
            if read_from_azure:
                self.conus404_ds = read_conus404_dataset_from_azure()

                self.conus404_da_dict = {}
                for var in self.input_variable_list:
                    # Get data for current variable
                    current_variable = self.conus404_ds[var]
              
                    # Save data indexed across the desired date/time range 
                    self.conus404_da_dict[var] = current_variable.loc[f"{self.start_dt:%Y-%m-%d %H:%M:%S}":f"{self.end_dt:%Y-%m-%d %H:%M:%S}"]

                    self._calculate_conus404_native_accum_precip_amount(var)
            else:
                # Here, we read native grid data from netCDF. This step is necessary to interpolate to a destination
                # grid using xesmf (interp_with_xesmf.py), since there are sometimes errors in accessing the Azure dataset
                # in the xesmf_env environment.
                self.conus404_native_accum_precip = self.read_precip_data_from_netcdf(temporal_res = 1, spatial_res = "native")

 
    ##### GETTER METHODS (CONUS404DataProcessor) #####
    def get_precip_data(self, temporal_res = "native", spatial_res = "native", load = False):
        match spatial_res:
            case "dest_grid":
                if (not self.DEST_GRID_FLAG): 
                    print(f"No data at {spatial_res} spatial resolution")
                    return
                data_array = self.read_precip_data_from_netcdf(temporal_res = temporal_res, spatial_res = spatial_res) 
            case "native":
                match temporal_res:
                    case "native":
                        data_array = self.conus404_native_accum_precip 
                    case _:
                        if self.DEST_GRID_FLAG: # Read from existing netCDFs because we haven't loaded data from Azure
                            data_array = self.read_precip_data_from_netcdf(temporal_res = temporal_res, spatial_res = spatial_res) 
                        else: # Data will have been loaded from Azure or from native grid netCDFs, so proceed normally
                            data_array = self._calculate_conus404_accum_precip_amount(time_period_hours = temporal_res)
            case _:
                print(f"No data at {spatial_res} spatial resolution; use 'native' or 'dest_grid'")
                return

        if load:
            print(f"Loading precip data for {self.data_name}")
            data_array.load()

        return data_array
    
    def how_to_get_precip_data(self):
        print("**** HOW TO GET CONUS404 PRECIP DATA:")
        print("Call get_precip_data(spatial_res = 'native', temporal_res = 'native', load = False)")
        print("The argument spatial_res can be 'dest_grid' or 'native'")
        print("The argument temporal_res can be 'native' or an integer representing the desired temporal resolution in hours")

    ##### PRIVATE METHODS (CONUS404DataProcessor) #####
    def _calculate_conus404_native_accum_precip_amount(self, var_name):
        match var_name:
            case "PREC_ACC_NC":
                print(f"Calculating accumulated {self.data_name} precipitation from {var_name}")

                # For CONUS404 precip, standardize DataArray attributes
                accum_precip_data_array = self.conus404_da_dict[var_name]
                add_attributes_to_data_array(accum_precip_data_array,
                                             short_name = f"{self.native_temporal_res:02d}-hour precipitation",
                                             long_name = f"Precipitation accumulated over the prior {self.native_temporal_res} hour(s)",
                                             units = "mm",
                                             interval_hours = self.native_temporal_res) 
                accum_precip_data_array.name = utils.accum_precip_var_name

                # Change the name of the time dimension from "time" to "period_end_time"
                accum_precip_data_array = accum_precip_data_array.rename({utils.time_dim_str: utils.period_end_time_dim_str}) 

                self.conus404_native_accum_precip = accum_precip_data_array 
            case _:
                pass

    # Sum totals at the native CONUS404 spatial resolution to derive totals over longer time periods
    # (3, 6, 12, 24 hours...).
    def _calculate_conus404_accum_precip_amount(self, time_period_hours = 3):
        time_step = int(time_period_hours/self.native_temporal_res)
        raw_data = self.conus404_native_accum_precip
        roller = raw_data.rolling({self.period_end_time_dim_str: time_step})
        accum_precip_data_array = roller.sum()[(time_step - 1)::time_step,:,:]

        add_attributes_to_data_array(accum_precip_data_array,
                                     short_name = f"{time_period_hours:02d}-hour precipitation",
                                     long_name = f"Precipitation accumulated over the prior {time_period_hours} hour(s)",
                                     units = self.conus404_native_accum_precip.units,
                                     interval_hours = time_period_hours)

        return accum_precip_data_array

    # Use cdo command line utility to interpolate from the native CONUS404 Lambert Conformal grid
    # to an output rectilinear grid. Using cdo requires that there exist netCDF files containing data
    # at the native CONUS404 resolution and at the same cadence (e.g. daily files) as the desired output.
    def _spatially_interpolate_conus404_to_dest_grid(self):
        if (not self.DEST_GRID_FLAG): 
            print("Destination grid flag not set; not spatially interpolating to destination grid")
            return 
        print(f"Spatially interpolating {self.data_name} data to {self.dest_grid_name} destination data grid")

        # Define template file whose grid we'll interpolate to
        output_grid_string = set_grid_name_for_file_names(self.dest_grid_name)
        template_file = f"{output_grid_string}.nc"
        template_fpath = os.path.join(self.netcdf_dir, "TemplateGrids", template_file)
        
        # If 24-hour precip timestamps are not valid at hour=00z, add a string to the directory and
        # file names denoting the hour at which 24-hour accumulated data starts (e.g., if 24-hour periods are from 12z-12z)
        hour_span_timestamp = ""
        if (self.dest_temporal_res == 24) and (self.end_dt.hour != 0):
            hour_span_timestamp = f".{self.end_dt.hour:02d}z-{self.end_dt.hour:02d}z"

        # Input data directory (data on native Replay grid) but at the desired temporal resolution (e.g., 24-hour precip)
        input_grid_name = set_grid_name_for_file_names("native")
        input_dir = f"{self.data_name}.{input_grid_name}.{self.dest_temporal_res:02d}_hour_precipitation{hour_span_timestamp}"
        input_dir = os.path.join(self.netcdf_dir, input_dir)
        if (not os.path.exists(input_dir)):
            print(f"Error: Input directory {input_dir} containing netCDF files on {self.data_name} native_grid does not exist")
            return
        
        # Collect list of input files at native grid resolution
        valid_daily_dt_list = construct_daily_datetime_list(self.start_dt, self.end_dt)
        native_grid_file_list = []
        for dtime in valid_daily_dt_list:
            fname = f"CONUS404.{input_grid_name}.{self.dest_temporal_res:02d}_hour_precipitation.{dtime:%Y%m%d}{hour_span_timestamp}.nc"
            fpath = os.path.join(input_dir, fname)
            if (not os.path.exists(fpath)):
                print(f"Warning: Input file path {fpath} does not exist; not including in input file list")
                continue
            native_grid_file_list.append(fpath)

        if (len(native_grid_file_list) == 0):
            print(f"Error: No files found in {self.data_name} native grid directory {input_dir}")
            return

        # Create output directory
        output_dir = f"{self.data_name}.{output_grid_string}.{self.dest_temporal_res:02d}_hour_precipitation{hour_span_timestamp}" 
        output_dir = os.path.join(self.netcdf_dir, output_dir)
        if (not os.path.exists(output_dir)):
            print(f"Creating directory {output_dir}")
            os.mkdir(output_dir)
            
        # Set CDO interpolation type flag
        cdo_interp_type = utils.set_cdo_interpolation_type(self.interp_method) 

        # Loop through each file, interpolating using cdo command line utility
        for input_fpath in native_grid_file_list:
            print(f"Interpolating {input_fpath} from {self.data_name} native grid to {self.dest_grid_name} grid")
            dstr = os.path.basename(input_fpath).split(".")[3]
            output_file = f"{self.data_name}.{output_grid_string}.{self.dest_temporal_res:02d}_hour_precipitation.{dstr}{hour_span_timestamp}.nc" 
            output_fpath = os.path.join(output_dir, output_file)
            cdo_cmd = f"cdo -P 8 {cdo_interp_type},{template_fpath} {input_fpath} {output_fpath}"
            print(f"Executing: {cdo_cmd}")
            os.system(cdo_cmd)

    ##### PUBLIC METHODS (CONUS404DataProcessor) #####
    def read_precip_data_from_netcdf(self, temporal_res = 24, spatial_res = "native"):
        # If 24-hour precip timestamps are not valid at hour=00z, add a string to the directory and
        # file names denoting the hour at which 24-hour accumulated data starts (e.g., if 24-hour periods are from 12z-12z)
        hour_span_timestamp = ""
        if (self.dest_temporal_res == 24) and (self.end_dt.hour != 0):
            hour_span_timestamp = f".{self.end_dt.hour:02d}z-{self.end_dt.hour:02d}z"

        # Construct list of CONUS404 netCDF files to read into xarray Dataset
        input_file_list = []
        if (spatial_res == "native"):
            output_grid_string = set_grid_name_for_file_names("native")
        elif (spatial_res == "dest_grid"):
            output_grid_string = set_grid_name_for_file_names(self.dest_grid_name)
        input_dir = os.path.join(utils.data_nc_dir, f"{self.data_name}.{output_grid_string}.{temporal_res:02d}_hour_precipitation{hour_span_timestamp}")
        for dtime in construct_daily_datetime_list(self.start_dt, self.end_dt): 
            fname = f"{self.data_name}.{output_grid_string}.{temporal_res:02d}_hour_precipitation.{dtime:%Y%m%d}{hour_span_timestamp}.nc"
            fpath = os.path.join(input_dir, fname)
            if (not os.path.exists(fpath)):
                print(f"Warning: Input file path {fpath} does not exist; not including in input file list")
                continue
            input_file_list.append(fpath)
        
        if (len(input_file_list) == 0):
            print(f"Error: No files found in input directory {input_dir}")
            sys.exit(1)

        # Open dataset
        print("Reading data") 
        ds = xr.open_mfdataset(input_file_list)
        da = ds[f"precipitation_{temporal_res:02d}_hour"]

        # Select data in specific date range
        return da.loc[f"{self.start_dt:%Y-%m-%d %H:%M:%S}":f"{self.end_dt:%Y-%m-%d %H:%M:%S}"]

    def write_precip_data_to_netcdf(self, spatial_res = "native", temporal_res = "native", file_cadence = "day", testing = False):
        if (spatial_res == "native"):
            # Get data to write
            full_data_array = self.get_precip_data(temporal_res = temporal_res, load = True)

            if (temporal_res == "native"):
                output_var_name = "precipitation"
                formatted_short_name = format_short_name(full_data_array)
            else:
                output_var_name = f"precipitation_{temporal_res:02d}_hour"
                formatted_short_name = f"{temporal_res:02d}_hour_precipitation"
            full_data_array.name = output_var_name
            
            # Construct file directory
            output_grid_string = set_grid_name_for_file_names("native")
            fname_prefix = f"{self.data_name}.{output_grid_string}.{formatted_short_name}" 
            dir_name = os.path.join(self.netcdf_dir, fname_prefix)
            
            # Construct timestamp format
            if (temporal_res == "native"):
                timestamp_format = "%Y%m%d.%H%M"
            elif (temporal_res < 24):
                timestamp_format = "%Y%m%d.%H"
            else:
                timestamp_format = "%Y%m%d"
            
            write_data_array_to_netcdf(full_data_array,
                                       output_var_name,
                                       dir_name,
                                       fname_prefix,
                                       timestamp_format,
                                       temporal_res = temporal_res,
                                       file_cadence = file_cadence,
                                       testing = testing)
        elif (spatial_res == "dest_grid") or (spatial_res == "model"):
            # This method will do the interpolation via cdo, so it writes to netCDF as part of that process.
            self._spatially_interpolate_conus404_to_dest_grid()
        else:
            print(f"No data at {spatial_res} spatial resolution; use 'native', 'dest_grid', or 'model'")
            return

class NestedReplayDataProcessor(object):
    def __init__(self,
                 start_dt_str,
                 end_dt_str,
                 data_name = "NestedReplay",
                 native_data_dir = os.path.join(utils.data_nc_dir, "NestedReplay.NativeGrid.01_hour_precipitation"),
                 native_temporal_res = 1, 
                 DEST_GRID_FLAG = False, # Set to True to spatially interpolate to destination grid 
                 dest_temporal_res = 24, # Temporal resolution of data we want to spatially interpolate to destination grid
                 dest_grid_name = "AORC",
                 interp_method = "linear",
                 region = "CONUS",
                 replay_segment = "corrector",
                 user_dir = "bbasarab"):
 
        # Process start and end datetime strings into all the date/time info we'll need
        self.start_dt_str = start_dt_str
        self.end_dt_str = end_dt_str
        self.start_dt = check_model_valid_dt_format(self.start_dt_str, resolution = 1)
        self.end_dt = check_model_valid_dt_format(self.end_dt_str, resolution = 1)
        self.first_day_dt = dt.datetime(self.start_dt.year, self.start_dt.month, self.start_dt.day)
        self.final_day_dt = dt.datetime(self.end_dt.year, self.end_dt.month, self.end_dt.day)

        # Set time dimension names
        self.time_dim_name = utils.time_dim_str 
        self.period_begin_time_dim_str = utils.period_begin_time_dim_str 
        self.period_end_time_dim_str = utils.period_end_time_dim_str

        # Set native temporal resolution and flag indicating whether there is a destination
        # grid to interpolate to. 
        self.native_temporal_res = native_temporal_res
        self.DEST_GRID_FLAG = DEST_GRID_FLAG
 
        # File reading/writing and plotting
        self.data_name = data_name
        self.native_data_dir = native_data_dir 
        self.region = region
        self.replay_segment = replay_segment
        self.user_dir = user_dir
        self.home_dir = os.path.join("/home", self.user_dir)
        self.data_dir = os.path.join("/data", self.user_dir) 
        self.plot_output_dir = os.path.join(self.home_dir, "plots") 
        self.netcdf_dir = os.path.join(self.data_dir, "netcdf")
        
        # Read native precipitation data that we will process 
        self._construct_valid_dt_lists()
        self._construct_input_file_list()
        self._read_input_files()
        
        # Extract hourly accumulated precipitation amounts 
        self._calculate_nested_replay_native_accum_precip_amount()

        # If DEST_GRID_FLAG = True, define the parameters of the separate grid to which we'll interpolate. 
        if self.DEST_GRID_FLAG:
            self.dest_grid_name = dest_grid_name
            self.dest_temporal_res = dest_temporal_res
            self.interp_method = interp_method
            self._spatially_interpolate_nested_replay_to_dest_grid()

    ##### GETTER METHODS (NestedReplayDataProcessor) #####
    def get_precip_data(self, spatial_res = "native", temporal_res = "native", load = False):
        if (spatial_res == "dest_grid") and not(self.DEST_GRID_FLAG):
            print(f"No data at {spatial_res} spatial resolution")
            return
        elif (spatial_res == "native") and (temporal_res == "native"):
            data_array = self.nested_replay_native_accum_precip
        else:
            data_array = self._calculate_nested_replay_accum_precip_amount(time_period_hours = temporal_res, spatial_res = spatial_res)
 
        if load:
            print(f"Loading precip data for {self.data_name}")
            data_array.load()

        return data_array

    def how_to_get_precip_data(self):
        print("**** HOW TO GET NESTED REPLAY PRECIP DATA:")
        print("Call get_precip_data(spatial_res = 'native', temporal_res = 'native', load = False)")
        print("The argument spatial_res can be 'dest_grid' or 'native'")
        print("The argument temporal_res can be 'native' or an integer representing the desired temporal resolution in hours")

    ##### PRIVATE METHODS (NestedReplayDataProcessor) #####
    def _construct_valid_dt_lists(self):
        # Construct datetimes valid at end of each period (this is how AORC data are presented)
        current_dt = self.start_dt
        self.valid_dt_list = [current_dt]
        while (current_dt != self.end_dt):
            current_dt += dt.timedelta(hours = self.native_temporal_res)
            self.valid_dt_list.append(current_dt)

        # Now construct list of valid days (PSL AORC files separated by day)
        self.daily_dt_list = list(dateutil.rrule.rrule(dateutil.rrule.DAILY, dtstart = self.first_day_dt, until = self.final_day_dt))

    def _construct_input_file_list(self):
        self.input_file_list = []
        for valid_dt in self.daily_dt_list:
            file_base_pattern = f"{self.data_name}.NativeGrid.accum_precip.nest.{self.replay_segment}.{valid_dt:%Y%m%d}.nc"
            file_path = os.path.join(self.native_data_dir, file_base_pattern)

            # Proceed to next file if a file doesn't exist (but need to handle any data gaps correctly)
            if (not os.path.exists(file_path)):
                print(f"Warning: Input file path {file_path} does not exist")
                continue 

            self.input_file_list.append(file_path)
            
    def _read_input_files(self):
        print(f"Reading {self.data_name} data from netCDF files into xarray Dataset")
        self.xr_dataset = xr.open_mfdataset(self.input_file_list)

    def _calculate_nested_replay_native_accum_precip_amount(self):
        print(f"Calculating accumulated {self.data_name} precipitation")

        # Use precip rate data directly, because it's average hourly precip rate (mm/hr) over the prior hour 
        accum_precip_data_array = self.xr_dataset["accum_precip"].loc[f"{self.start_dt:%Y-%m-%d %H:%M:%S}":f"{self.end_dt:%Y-%m-%d %H:%M:%S}"]

        # Change the name of the time dimension from "time" to "period_end_time"
        accum_precip_data_array = accum_precip_data_array.rename({utils.time_dim_str: utils.period_end_time_dim_str,
                                                                  "grid_xt": "lon",
                                                                  "grid_yt": "lat"}) 
       
        # Add attributes to the accumulated precip data array
        add_attributes_to_data_array(accum_precip_data_array,
                                     short_name = f"{self.native_temporal_res:02d}-hour precipitation",
                                     long_name = f"Precipitation accumulated over the prior {self.native_temporal_res} hour(s)",
                                     units = "mm",
                                     interval_hours = self.native_temporal_res) 

        self.nested_replay_native_accum_precip = accum_precip_data_array 

    # Sum totals at the NestedReplay temporal resolution to derive totals over longer time periods
    # (3, 6, 12, 24 hours...).
    def _calculate_nested_replay_accum_precip_amount(self, time_period_hours = 3, spatial_res = "native"):
        if (spatial_res == "native"):
            raw_data = self.nested_replay_native_accum_precip
        else:
            if not(self.DEST_GRID_FLAG): 
                print(f"No data at {spatial_res} spatial resolution")
                return
            raw_data = self.precip_dest_grid

        time_step = int(time_period_hours/self.native_temporal_res)
        roller = raw_data.rolling({self.period_end_time_dim_str: time_step})
        accum_precip_data_array = roller.sum()[(time_step - 1)::time_step,:,:]

        add_attributes_to_data_array(accum_precip_data_array,
                                     short_name = f"{time_period_hours:02d}-hour precipitation",
                                     long_name = f"Precipitation accumulated over the prior {time_period_hours} hour(s)",
                                     units = self.nested_replay_native_accum_precip.units,
                                     interval_hours = time_period_hours)

        return accum_precip_data_array

    # TODO: Implement ability to interpolate data to the destination grid resolution,
    # which may not be the same as the Nested Replay native temporal resolution of 1 hour.
    def _spatially_interpolate_nested_replay_to_dest_grid(self):
        if (not self.DEST_GRID_FLAG): 
            print("No destination grid specified; not spatially interpolating")
            return 
        print(f"Spatially interpolating {self.data_name} data to {self.dest_grid_name}")

        # Ensure we select Nested Replay data at the native temporal resolution to interpolate.
        # Calculation of accmulated amounts at different temporal resolutions is handled by
        # _calculate_nested_replay_accum_precip_amount.
        nested_replay_precip_to_interpolate = self.get_precip_data(temporal_res = "native").copy()

        # Get destination data array by instantiating the appropriate class
        match self.dest_grid_name:
            case "AORC":
                dest_grid_processor = AorcDataProcessor(self.start_dt_str,
                                                        self.end_dt_str,
                                                        DEST_GRID_FLAG = False,
                                                        region = self.region)
            case "StageIV":
                print("Error: Interpolation to StageIV grid not yet implemented")
                return 
            case _:
                print(f"Error: Interpolation to {self.dest_grid_name} grid not yet implemented")
                return

        # Spatially interpolate Nested Replay data to output grid 
        self.dest_data_array = dest_grid_processor.get_precip_data(spatial_res = "native", temporal_res = "native")
        self.precip_dest_grid = spatially_interpolate_using_interp_like(nested_replay_precip_to_interpolate,
                                                                        self.dest_data_array,
                                                                        interp_method = self.interp_method, 
                                                                        correct_small_negative_values = True)   

    # Use cdo command line utility to interpolate from the native NestedReplay grid
    # Using cdo requires that there exist netCDF files containing data at the native NestedReplay
    # resolution and at the same cadence (e.g. daily files) as the desired output.
    def _spatially_interpolate_nested_replay_to_dest_grid_using_cdo(self):
        if (not self.DEST_GRID_FLAG): 
            print("Destination grid flag not set; not spatially interpolating to destination grid")
            return 
        print(f"Spatially interpolating {self.data_name} data to {self.dest_grid_name} destination data grid")

        # Define template file whose grid we'll interpolate to
        output_grid_string = set_grid_name_for_file_names(self.dest_grid_name)
        template_file = f"{output_grid_string}.nc"
        template_fpath = os.path.join(self.netcdf_dir, "TemplateGrids", template_file)

        # Input data directory (data on native Replay grid) but at the desired temporal resolution (e.g., 24-hour precip)
        input_grid_name = set_grid_name_for_file_names("native")
        input_dir = f"{self.data_name}.{input_grid_name}.{self.dest_temporal_res:02d}_hour_precipitation"
        input_dir = os.path.join(self.netcdf_dir, input_dir)
        if (not os.path.exists(input_dir)):
            print(f"Error: Input directory containing netCDF files on {self.data_name} native_grid does not exist")
            return
        
        # Collect list of input files at native grid resolution
        valid_daily_dt_list = construct_daily_datetime_list(self.start_dt, self.end_dt)
        native_grid_file_list = []
        for dtime in valid_daily_dt_list:
            fname = f"NestedReplay.{input_grid_name}.{self.dest_temporal_res:02d}_hour_precipitation.{dtime:%Y%m%d}.nc"
            fpath = os.path.join(input_dir, fname)
            if (not os.path.exists(fpath)):
                print(f"Warning: Input file path {fpath} does not exist; not including in input file list")
                continue
            native_grid_file_list.append(fpath)

        if (len(native_grid_file_list) == 0):
            print(f"Error: No files found in {self.data_name} native grid directory {input_dir}")
            return

        # Create output directory
        output_dir = f"{self.data_name}.{output_grid_string}.{self.dest_temporal_res:02d}_hour_precipitation" 
        output_dir = os.path.join(self.netcdf_dir, output_dir)
        if (not os.path.exists(output_dir)):
            print(f"Creating directory {output_dir}")
            os.mkdir(output_dir)
        
        # Set CDO interpolation type flag
        cdo_interp_type = utils.set_cdo_interpolation_type(self.interp_method) 
            
        # Loop through each file, interpolating using cdo command line utility
        for input_fpath in native_grid_file_list:
            print(f"Interpolating {input_fpath} from {self.data_name} native grid to {self.dest_grid_name} grid")
            dstr = os.path.basename(input_fpath).split(".")[3]
            output_file = f"{self.data_name}.{output_grid_string}.{self.dest_temporal_res:02d}_hour_precipitation.{dstr}.nc" 
            output_fpath = os.path.join(output_dir, output_file)
            cdo_cmd = f"cdo -P 8 {cdo_interp_type},{template_fpath} {input_fpath} {output_fpath}"
            print(f"Executing: {cdo_cmd}")
            os.system(cdo_cmd)

    ##### PUBLIC METHODS (NestedReplayDataProcessor) #####
    def write_precip_data_to_netcdf(self, temporal_res = "native", spatial_res = "native", file_cadence = "day", testing = False):
        if (spatial_res == "dest_grid") and (not self.DEST_GRID_FLAG):
            print(f"No data at {spatial_res} spatial resoluation to write to netCDF")
            return
 
        # Get data to write
        full_data_array = self.get_precip_data(temporal_res = temporal_res, spatial_res = spatial_res, load = True)

        if (temporal_res == "native"):
            output_var_name = "precipitation_01_hour"
            formatted_short_name = "01_hour_precipitation" 
        else:
            output_var_name = f"precipitation_{temporal_res:02d}_hour"
            formatted_short_name = f"{temporal_res:02d}_hour_precipitation"
        full_data_array.name = output_var_name

        if (spatial_res == "native"):
            output_grid_string = set_grid_name_for_file_names(spatial_res)
        elif (spatial_res == "dest_grid"):
            output_grid_string = set_grid_name_for_file_names(self.dest_grid_name)
        
        # Construct file directory
        # Output corrector segment data with dir and file names 'NestedReplay'
        # Output predictor segment data with dir and file names 'NestedReplayPredictor'
        output_data_name_str = f"{self.data_name}"
        if (self.replay_segment == "predictor"): 
            output_data_name_str += self.replay_segment.title() 
        
        dir_name = f"{output_data_name_str}.{output_grid_string}.{formatted_short_name}" 
        dir_name = os.path.join(self.netcdf_dir, dir_name) 
        fname_prefix = f"{output_data_name_str}.{output_grid_string}.{formatted_short_name}" 
 
        # Construct timestamp format
        if (temporal_res == "native"):
            timestamp_format = "%Y%m%d.%H%M"
        elif (temporal_res < 24):
            timestamp_format = "%Y%m%d.%H"
        else:
            timestamp_format = "%Y%m%d"
        
        write_data_array_to_netcdf(full_data_array,
                                   output_var_name,
                                   dir_name,
                                   fname_prefix,
                                   timestamp_format,
                                   temporal_res = temporal_res,
                                   file_cadence = file_cadence,
                                   testing = testing)

class StageIVDataProcessor(object):
    def __init__(self,
                 start_dt_str,
                 end_dt_str,
                 obs_name = "StageIV",
                 native_data_dir = "/data/bbasarab/grib/StageIV",
                 obs_temporal_res = 1, 
                 DEST_GRID_FLAG = False, # Set to True to spatially interpolate to destination grid 
                 dest_temporal_res = 24, # Temporal resolution of data we want to spatially interpolate to destination grid
                 dest_grid_name = "Replay",
                 interp_method = "linear",
                 region = "CONUS",
                 user_dir = "bbasarab"):

        # Process start and end datetime strings into all the date/time info we'll need
        self.DEST_GRID_FLAG = DEST_GRID_FLAG
        self.obs_temporal_res = obs_temporal_res
        self.start_dt_str = start_dt_str
        self.end_dt_str = end_dt_str
        self.start_dt = check_model_valid_dt_format(self.start_dt_str, resolution = 1)
        self.end_dt = check_model_valid_dt_format(self.end_dt_str, resolution = 1)
        self.first_day_dt = dt.datetime(self.start_dt.year, self.start_dt.month, self.start_dt.day)
        self.final_day_dt = dt.datetime(self.end_dt.year, self.end_dt.month, self.end_dt.day)

        # Set time dimension names
        self.time_dim_name = utils.time_dim_str 
        self.period_begin_time_dim_str = utils.period_begin_time_dim_str 
        self.period_end_time_dim_str = utils.period_end_time_dim_str
 
        # File reading/writing and plotting
        self.obs_name = obs_name
        self.native_data_dir = native_data_dir
        self.region = region
        self.user_dir = user_dir
        self.home_dir = os.path.join("/home", self.user_dir)
        self.data_dir = os.path.join("/data", self.user_dir) 
        self.plot_output_dir = os.path.join(self.home_dir, "plots") 
        self.netcdf_output_dir = os.path.join(self.data_dir, "netcdf")        

        # Read native precipitation data that we will process 
        self._construct_valid_dt_lists()
        self._read_input_files()

        # Extract hourly accumulated precipitation amounts (native StageIV data are hourly) 
        self._calculate_stageiv_native_accum_precip_amount()

        # Set an accompanying dest grid to interpolate to
        if self.DEST_GRID_FLAG:
            self.dest_grid_name = dest_grid_name
            self.dest_grid_native_temporal_res = get_dataset_native_temporal_res(self.dest_grid_name) # Native temporal resolution of destination grid
            self.dest_temporal_res = dest_temporal_res
            self.interp_method = interp_method
            self._spatially_interpolate_stageiv_to_dest_grid()

    ##### GETTER METHODS (StageIVDataProcessor) #####
    def get_precip_data(self, spatial_res = "native", temporal_res = "native", load = False):
        match spatial_res:
            case "dest_grid":
                if (not self.DEST_GRID_FLAG): 
                    print(f"No data at {spatial_res} spatial resolution")
                    return
                data_array = self._calculate_stageiv_accum_precip_amount(time_period_hours = temporal_res, spatial_res = spatial_res) 
            case "native":
                match temporal_res:
                    case "native":
                        data_array = self.stageiv_native_accum_precip 
                    case _:
                        data_array = self._calculate_stageiv_accum_precip_amount(time_period_hours = temporal_res)
            case _:
                print(f"No data at {spatial_res} spatial resolution; use 'native' or 'dest_grid'")
                return

        if load:
            print(f"Loading precip data for {self.obs_name}")
            data_array.load()

        return data_array
 
    def get_dest_grid_name(self):
        if (not self.DEST_GRID_FLAG):
            print("No accompanying dest grid data")
            return 
        return self.dest_grid_name

    def get_obs_name(self):
        return self.obs_name
 
    def how_to_get_precip_data(self):
        print("**** HOW TO GET StageIV PRECIP DATA:")
        print("Call get_precip_data(spatial_res = 'native', temporal_res = 'native', load = False)")
        print("The argument spatial_res can be 'dest_grid' or 'native'")
        print("The argument temporal_res can be 'native' or an integer representing the desired temporal resolution in hours")

    ##### PRIVATE METHODS (StageIVDataProcessor) #####
    def _construct_valid_dt_lists(self):
        # Construct datetimes valid at end of each period (this is how StageIV data are presented)
        current_dt = self.start_dt
        self.valid_dt_list = [current_dt]
        while (current_dt != self.end_dt):
            current_dt += dt.timedelta(hours = self.obs_temporal_res)
            self.valid_dt_list.append(current_dt)

        # Now construct list of valid days (PSL StageIV files separated by day)
        self.daily_dt_list = list(dateutil.rrule.rrule(dateutil.rrule.DAILY, dtstart = self.first_day_dt, until = self.final_day_dt))

        # Remove full days with any missing files for that day
        self.daily_dtimes_with_missing_data = [] 
        self.valid_dtimes_with_missing_data = []
        for valid_dt in self.valid_dt_list:
            if ( (valid_dt - dt.datetime(2020, 7, 28, 0)).total_seconds() <= 0):
                file_base_pattern = "ST4.{input_dtime:%Y%m%d%H}.01h"
            else:
                file_base_pattern = "st4_conus.{input_dtime:%Y%m%d%H}.01h.grb2"
            file_path = os.path.join(self.native_data_dir, file_base_pattern.format(input_dtime = valid_dt))

            # There are a fair amount of missing StageIV grib files. For now (as of 20260526),
            # exclude the entire day if there are any missing files for that day. 
            if (not os.path.exists(file_path)):
                if (valid_dt.hour == 0): # Hour-ending StageIV data valid at 00z belongs to the previous valid day
                    daily_dtime_with_missing_data = dt.datetime(valid_dt.year, valid_dt.month, valid_dt.day) - dt.timedelta(days = 1)
                else:
                    daily_dtime_with_missing_data = dt.datetime(valid_dt.year, valid_dt.month, valid_dt.day)
                print(f"Warning: Input file path {file_path} does not exist; not including any data from {daily_dtime_with_missing_data:%Y%m%d}")

                #self.valid_dtimes_with_missing_data.append(valid_dt) # This line excludes only specific missing valid times (not entire days with any missing data)
                self.valid_dtimes_with_missing_data += [daily_dtime_with_missing_data + dt.timedelta(hours = i) for i in range(1, 25)] 

                if (daily_dtime_with_missing_data not in self.daily_dtimes_with_missing_data):
                    self.daily_dtimes_with_missing_data.append(daily_dtime_with_missing_data)

        self._remove_valid_dtimes_with_missing_data()

    def _remove_valid_dtimes_with_missing_data(self):
        self.valid_dt_list = [dtime for dtime in self.valid_dt_list if dtime not in self.valid_dtimes_with_missing_data]
        self.daily_dt_list = [dtime for dtime in self.daily_dt_list if dtime not in self.daily_dtimes_with_missing_data]
            
    def _read_input_files(self):
        print(f"Reading {self.obs_name} data from grib format")

        precip_da_list = []
        self.input_file_list = []
        for d, dtime in enumerate(self.valid_dt_list):
            if ( (dtime - dt.datetime(2020, 7, 28, 0)).total_seconds() <= 0):
                file_base_pattern = "ST4.{input_dtime:%Y%m%d%H}.01h"
            else:
                file_base_pattern = "st4_conus.{input_dtime:%Y%m%d%H}.01h.grb2"
            file_path = os.path.join(self.native_data_dir, file_base_pattern.format(input_dtime = dtime))
            print(f"Input grib file path: {file_path}")
            self.input_file_list.append(file_path)
            
            grib_data = pygrib.open(file_path)
            grib_message = grib_data.select()[0]

            if (d == 0):
                lats, lons = grib_message.latlons()
                nx = lats.shape[0]
                ny = lats.shape[1]

            precip_data = grib_message.values
            precip_data = np.ma.masked_where(precip_data > 320, precip_data)
            precip_data = np.ma.filled(precip_data, np.nan)

            period_end_time = self.valid_dt_list[d]

            precip_da_single_time = xr.DataArray(precip_data,
                                                 dims = ["y", "x"],
                                                 coords = {
                                                           "lat": (["y", "x"], lats),
                                                           "lon": (["y", "x"], lons),
                                                          })

            # Add time dimension
            precip_da_single_time = precip_da_single_time.expand_dims(dim = {utils.period_end_time_dim_str: [period_end_time]})

            precip_da_list.append(precip_da_single_time)

        self.precip_da = xr.concat(precip_da_list, dim = utils.period_end_time_dim_str)

    # Calculate accumulated precip at the native temporal resolution of StageIV (i.e., hourly)
    # StageIV presents average precip rate in mm/hr over the prior hour, so just used these values directly to represent hourly accumulated precipitation 
    def _calculate_stageiv_native_accum_precip_amount(self):
        print(f"Calculating accumulated {self.obs_name} precipitation from prate")

        # Use precip rate data directly, because it's average hourly precip rate (mm/hr) over the prior hour 
        # accum_precip_data_array = self.xr_dataset["precrate"].loc[f"{self.start_dt:%Y-%m-%d %H:%M:%S}":f"{self.end_dt:%Y-%m-%d %H:%M:%S}"] # /Projects/BIL area with time offset
        accum_precip_data_array = self.precip_da.loc[f"{self.start_dt:%Y-%m-%d %H:%M:%S}":f"{self.end_dt:%Y-%m-%d %H:%M:%S}"]

        # Add attributes to the accumulated precip data array
        add_attributes_to_data_array(accum_precip_data_array,
                                     short_name = f"{self.obs_temporal_res:02d}-hour precipitation",
                                     long_name = f"Precipitation accumulated over the prior {self.obs_temporal_res} hour(s)",
                                     units = "mm",
                                     interval_hours = self.obs_temporal_res) 
        accum_precip_data_array.name = utils.accum_precip_var_name

        self.stageiv_native_accum_precip = accum_precip_data_array 

    # Sum totals at the StageIV temporal resolution to derive totals over longer time periods
    # (3, 6, 12, 24 hours...).
    def _calculate_stageiv_accum_precip_amount(self, time_period_hours = 3, spatial_res = "native"):
        if (spatial_res == "dest_grid"):
            time_step = int(time_period_hours/self.dest_temporal_res)
            raw_data = self.precip_dest_grid
        else:
            time_step = int(time_period_hours/self.obs_temporal_res)
            raw_data = self.stageiv_native_accum_precip
        roller = raw_data.rolling({self.period_end_time_dim_str: time_step})
        accum_precip_data_array = roller.sum()[(time_step - 1)::time_step,:,:]

        add_attributes_to_data_array(accum_precip_data_array,
                                     short_name = f"{time_period_hours:02d}-hour precipitation",
                                     long_name = f"Precipitation accumulated over the prior {time_period_hours} hour(s)",
                                     units = self.stageiv_native_accum_precip.units,
                                     interval_hours = time_period_hours)

        return accum_precip_data_array

    def _spatially_interpolate_stageiv_to_dest_grid(self):
        if (not self.DEST_GRID_FLAG): 
            print(f"No dest grid data; not spatially interpolating to {self.dest_grid} grid")
            return 
        print(f"Spatially interpolating {self.obs_name} data to {self.dest_grid} grid")

        stageiv_precip_to_interpolate = self.get_precip_data(spatial_res = "native",
                                                          temporal_res = self.dest_grid_native_temporal_res).copy()

        # Change longitude coordinates of StageIV to be a subset of [0, 360] to match Replay coordinates.
        if (self.dest_grid_name == "Replay"):
            stageiv_lons_0to360 = utils.longitude_to_0to360(stageiv_precip_to_interpolate["lon"].values)
            stageiv_precip_to_interpolate["lon"] = stageiv_lons_0to360 
            stageiv_precip_to_interpolate = stageiv_precip_to_interpolate.sortby("lon") 

        # Get destination data array by instantiating the appropriate class
        match self.dest_grid_name:
            case "AORC":
                dest_grid_processor = StageIVDataProcessor(self.start_dt_str,
                                                        self.end_dt_str,
                                                        DEST_GRID_FLAG = False,
                                                        region = self.region)
            case "Replay":
                start_dt_dest_grid = self.start_dt + dt.timedelta(hours = self.dest_grid_native_temporal_res - self.temporal_res)
                end_dt_dest_grid = self.end_dt
                dest_grid_processor = ReplayDataProcessor(start_dt_dest_grid.strftime("%Y%m%d.%H"),
                                                          end_dt_dest_grid.strftime("%Y%m%d.%H"),
                                                          DEST_GRID_FLAG = False,
                                                          region = self.region)
            case _:
                print(f"Error: Interpolation to {self.dest_grid_name} grid not yet implemented")
                return

        # FIXME: xarray's interp_like method may not work since StageIV lat/lons are 2D. May need to use cdo.
        self.dest_data_array = dest_grid_processor.get_precip_data(spatial_res = "native", temporal_res = self.dest_grid_native_temporal_res)
        self.precip_dest_grid = spatially_interpolate_using_interp_like(stageiv_precip_to_interpolate,
                                                                        self.dest_data_array,
                                                                        interp_method = self.interp_method, 
                                                                        correct_small_negative_values = True)   
    
    ##### PUBLIC METHODS (StageIVDataProcessor) #####
    def write_precip_data_to_netcdf(self, temporal_res = "native", spatial_res = "native", file_cadence = "day", testing = False):
        if (spatial_res == "dest_grid") and (not self.DEST_GRID_FLAG):
            print(f"No data at {spatial_res} spatial resoluation to write to netCDF")
            return 

        # Get data to write
        full_data_array = self.get_precip_data(temporal_res = temporal_res, spatial_res = spatial_res, load = True)

        if (temporal_res == "native"):
            output_var_name = "precipitation"
            formatted_short_name = format_short_name(full_data_array)
        else:
            output_var_name = f"precipitation_{temporal_res:02d}_hour"
            formatted_short_name = f"{temporal_res:02d}_hour_precipitation"
        full_data_array.name = output_var_name
        
        if (spatial_res == "native"):
            output_grid_string = set_grid_name_for_file_names(spatial_res)
        elif (spatial_res == "dest_grid"):
            output_grid_string = set_grid_name_for_file_names(self.dest_grid_name)

        # Construct file directory
        fname_prefix = f"{self.obs_name}.{output_grid_string}.{formatted_short_name}"
        dir_name = os.path.join(self.netcdf_output_dir, fname_prefix)
        
        # Construct timestamp format
        if (temporal_res == "native"):
            timestamp_format = "%Y%m%d.%H%M"
        elif (temporal_res < 24):
            timestamp_format = "%Y%m%d.%H"
        else:
            timestamp_format = "%Y%m%d"
        
        write_data_array_to_netcdf(full_data_array,
                                   output_var_name,
                                   dir_name,
                                   fname_prefix,
                                   timestamp_format,
                                   temporal_res = temporal_res,
                                   file_cadence = file_cadence,
                                   testing = testing)
