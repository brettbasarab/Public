import cartopy.crs as ccrs
import cartopy.feature as cfeature
import dataclasses
import datetime as dt
import dateutil
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import precip_data_processors as pdp
import precip_plotting_utilities as ppu
import pylab
import scipy
import sys
import utilities as utils
import xarray as xr
from typing import Any, Dict, Iterable, List, Mapping, Tuple

# TODO (both low priority):
    # Figure out the source of the "All-NaN slice encountered" warning in percentile calculations
    # Update plot names to use '.' notation

@dataclasses.dataclass
class StatsDataClass:
    threshold: float
    hits: int
    misses: int
    false_alarms: int
    correct_negatives: int
    total_events: int
    frequency_bias: float
    CSI: float
    ETS: float

# Keyword to evaluate FSS for varying radius, using a fixed amount threshold
evaluate_by_radius_kw_str = "by_radius"

# Keyword to evaluate FSS for varying radius, using a fixed ARI grid as threshold
evaluate_by_radius_ari_threshold_kw_str = "by_radius_ari_threshold"

# Keyword to evaluate FSS for varying amount thresholds 
evaluate_by_threshold_kw_str = "by_threshold"

# Keyword to evaluate FSS using varying ARI grids as thresholds
evaluate_by_ari_kw_str = "by_ari_grid"

# For a given region, return a list of the names of the datasets currently used
# in verification, as well as the dataset that will be considered truth (truth_data_name)
def map_region_to_data_names(region, verif_grid = utils.Replay_data_name, include_hrrr = False):
    if ("US" in region): 
        truth_data_name = utils.AORC_data_name # "AORC" 
        data_grid = utils.Replay_data_name # "Replay"
        data_names = [utils.AORC_data_name,
                      utils.CONUS404_data_name,
                      utils.ERA5_data_name,
                      utils.IMERG_data_name,
                      utils.Replay_data_name] # ["AORC", "CONUS404", "ERA5", "IMERG", "Replay"]
        if (verif_grid == utils.AORC_data_name): 
            data_grid = utils.AORC_data_name # "AORC"
            data_names = [utils.AORC_data_name,
                          utils.CONUS404_data_name,
                          utils.NestedReplay_data_name,
                          utils.Replay_data_name] # ["AORC", "CONUS404", "NestedReplay", "Replay"]
        if include_hrrr:
            data_names.append(utils.HRRR_data_name) # "HRRR"
    else: 
        truth_data_name = utils.IMERG_data_name # "IMERG"
        data_grid = utils.Replay_data_name # "Replay"
        data_names = [utils.IMERG_data_name,
                      utils.Replay_data_name,
                      utils.ERA5_data_name] # ["IMERG", "Replay", "ERA5"]

    return data_names, truth_data_name, data_grid

def how_to_inst_pvp_class():
  print('PrecipVerificationProcessor(\n'
        '    start_dt_str, end_dt_str,\n'
        '    LOAD_DATA = True, # If True, load data from netCDF, zarr, etc. files specified by input_format (otherwise use existing non-region-subsetted arrays)\n'
        '    loaded_non_subset_da_dict = None, # Dictionary of loaded but non-region-subsetted DataArrays\n'
        '    USE_EXTERNAL_DA_DICT = False, # If True, use data from a DataArray dictionary specified by external_da_dict; if False, load data directly specified by input_format\n'
        '    IS_STANDARD_INPUT_DICT = True, # If True, assumes that external_da_dict is a dictionary of the form {data_name_1: da_1, ...,data_name_N: da_N}\n'  
        '    LAT_LON_2D = False, # If True, latitude and longitude coordinates are 2D (e.g, for a curvilinear grid). Lat and lon are function of other (e.g., x,y) dimensions\n'
        '    external_da_dict = None, # Dictionary of precalculated data arrays for which to make plots (only used if USE_EXTERNAL_DA_DICT = True)\n'
        '    data_names = ["AORC", "IMERG", "Replay", "ERA5"],\n'
        '    truth_data_name = "AORC", # Dataset that is considered truth (i.e., observations) for this particular verification\n'
        '    data_grid = "Replay", # Dataset whose grid verification is performed on\n'
        '    input_format = "netcdf",\n'
        '    model_var_list = ["prateb_ave"],\n'
        '    region = "Global",\n'
        '    region_info = None,\n'
        '    temporal_res = 3,\n'
        '    thresholds = [0.25, 1.0, 10.0], #, 25.0, 50.0], # Thresholds in mm\n'
        '    percentiles = [95, 99], # Percentiles to calculate\n'
        '    input_dir = None, # Top-level location of input data (nc or zarr) to use in verification)\n' 
        '    output_dir = None, # Top-level location for output data (stats and plots)\n'
        '    user_dir = None,\n' 
        '    poster = False)') 

class PrecipVerificationProcessor(object):
    def __init__(self, start_dt_str, end_dt_str,
                 LOAD_DATA = True, # If True, load data from netCDF, zarr, etc. files specified by input_format (otherwise use existing non-region-subsetted arrays)
                 loaded_non_subset_da_dict = None, # Dictionary of loaded but non-region-subsetted DataArrays
                 USE_EXTERNAL_DA_DICT = False, # If True, use data from an DataArray dictionary specified by external_da_dict; if False, load data directly specified by input_format
                 IS_STANDARD_INPUT_DICT = True, # If True, assumes that external_da_dict is a dictionary of the form {data_name_1: da_1, ...,data_name_N: da_N} 
                 LAT_LON_2D = False, # If True, latitude and longitude coordinates are 2D (e.g, for a curvilinear grid). Lat and lon are function of other (e.g., x,y) dimensions
                 external_da_dict = None, # Dictionary of precalculated data arrays for which to make plots (only used if USE_EXTERNAL_DA_DICT = True)
                 data_names = ["AORC", "IMERG", "Replay", "ERA5"],
                 truth_data_name = "AORC", # Dataset that is considered truth (i.e., observations) for this particular verification
                 data_grid = "Replay", # Dataset whose grid verification is performed on
                 grid_cell_size = None,
                 input_format = "netcdf",
                 model_var_list = ["prateb_ave"],
                 region = "Global",
                 region_info = None,
                 temporal_res = 3,
                 thresholds = [0.25, 1.0, 10.0], #, 25.0, 50.0], # Thresholds in mm
                 percentiles = [95, 99], # Percentiles to calculate
                 input_dir = None, # Top-level location of input data (nc or zarr) to use in verification) 
                 output_dir = None, # Top-level location for output data (stats and plots)
                 user_dir = None, 
                 poster = False): 

        self.LOAD_DATA = LOAD_DATA
        if not(self.LOAD_DATA):
            print("Non-region subsetted DataArrays already loaded")
            self.loaded_non_subset_da_dict = loaded_non_subset_da_dict
        else:
            print("Non-region subsetted DataArrays NOT already loaded; will load data accordingly")
        self.data_grid = data_grid
        self.data_grid_name = pdp.set_grid_name_for_file_names(self.data_grid)
        if (grid_cell_size is not None):
            self.grid_cell_size = grid_cell_size
        elif (self.data_grid == "Replay"):
            self.grid_cell_size = utils.replay_grid_cell_size
        elif(self.data_grid == "AORC"):
            self.grid_cell_size = utils.aorc_grid_cell_size
        else:
            self.grid_cell_size = 0.25
        self.native_grid_name = pdp.set_grid_name_for_file_names("Native")
        self.model_var_list = model_var_list
        self.temporal_res = temporal_res
        self.thresholds = thresholds
        self.percentiles = percentiles
        self.num_thresholds = len(self.thresholds)
        self.variable_plot_limits = ppu.variable_plot_limits("accum_precip", temporal_res = self.temporal_res)
        self.variable_pctl_plot_limits = ppu.variable_pctl_plot_limits("accum_precip", temporal_res = self.temporal_res)

        # Construct input directory
        if (user_dir is None):
            self.user_dir = "bbasarab"
        else: 
            self.user_dir = user_dir
        print(f"User directory: {self.user_dir}")

        if (input_dir is None):
            self.input_dir = os.path.join("/data", self.user_dir)
            self._set_input_format(input_format, name_input_dir_by_input_format = True)
        else:
            self.input_dir = input_dir 
            self._set_input_format(input_format, name_input_dir_by_input_format = False)
        print(f"Input data directory: {self.input_dir}")
        
        # Construct output directories
        if (output_dir is None):
            self.output_dir = os.path.join("/home", self.user_dir)
        else:
            self.output_dir = output_dir

        self.plot_output_dir = os.path.join(self.output_dir, "plots")
        if not(os.path.exists(self.plot_output_dir)):
            ret = os.makedirs(self.plot_output_dir)
            if (ret != 0):
                print(f"Error: Could not create plot output directory {self.plot_output_dir}")
                sys.exit(1) 
        print(f"Plots output directory: {self.plot_output_dir}")

        self.stats_output_dir = os.path.join(self.output_dir, "stats")
        if not(os.path.exists(self.stats_output_dir)):
            ret = os.makedirs(self.stats_output_dir)
            if (ret != 0):
                print(f"Error: Could not create plot output directory {self.stats_output_dir}")
                sys.exit(1) 
        print(f"Stats output directory: {self.stats_output_dir}")

        # Set pertinent region info (region lat/lon extents; whether region spans meridian, etc.)
        self._set_region_info(region, region_info)

        # If plots are for a poster, increase the fontsize of plot titles, axes labels, etc. 
        self.poster_font_increase = 0
        if poster:
            self.poster_font_increase = 2

        # Create lists of valid datetimes at the desired temporal resolution and at a daily cadence
        self.start_dt_str = start_dt_str
        self.end_dt_str = end_dt_str
        print(f"******* Instance of PrecipVerificationProcessorClass *******")
        print(f"Region: {self.region}")
        print(f"Start datetime: {self.start_dt_str}")
        print(f"End datetime: {self.end_dt_str}")
        print(f"Temporal resolution (hours): {self.temporal_res}")
        self._construct_valid_dt_lists()

        # Set data names list, truth data name, and related variables
        self.data_names = data_names
        self.truth_data_name = truth_data_name
        self._reorder_data_names_with_truth_data_first()
        self.data_names_str = "".join(f"{data_name}." for data_name in self.data_names)
        self.num_datasets = len(self.data_names)
        print(f"Data name list: {self.data_names}")
        print(f"Truth data name: {self.truth_data_name}")
        
        # Set standard dimension names
        self.time_dim_name = "period_end_time" 
        
        # Set contour pplot color maps, plot levels, and color lists
        self.plot_cmap, self.plot_levels, self.color_list = ppu.create_precip_plot_levels(temporal_res = self.temporal_res, version = 2)
        self.sum_plot_cmap, self.sum_plot_levels, self.sum_color_list = ppu.create_precip_plot_levels(temporal_res = self.get_num_valid_times(),
                                                                                                      version = 2)
        # Load data for verification 
        self.USE_EXTERNAL_DA_DICT = USE_EXTERNAL_DA_DICT
        self.IS_STANDARD_INPUT_DICT = IS_STANDARD_INPUT_DICT
        self.LAT_LON_2D = LAT_LON_2D 
        if not(self.USE_EXTERNAL_DA_DICT):
            # Read model and obs data, check that the data array shapes match, then load the data
            self._read_region_subset_and_load_data()
        else:
            print("**** USE_EXTERNAL_DA_DICT set; will perform any verification using external data arrays provided")
            if (external_da_dict is None):
                print("Error: USE_EXTERNAL_DA_DICT set but no external data arrays provided; can't perform verification")
                sys.exit(1)
            else:
                if self.IS_STANDARD_INPUT_DICT:
                    self.da_dict = {}
                    for data_name in self.data_names: # This loop ensures self.da_dict will maintain desired order of datasets (with truth data first)
                        if (data_name not in external_da_dict.keys()):
                            print(f"Error: Dataset {data_name} not in provided dictionary of external data arrays ({external_da_dict.keys()})")
                            sys.exit(1)
                        self._check_for_standard_dimensions(external_da_dict[data_name])
                        pdp.add_attributes_to_data_array(external_da_dict[data_name], interval_hours = self.temporal_res)
                        self.da_dict[data_name] = external_da_dict[data_name]
                else:
                    self.da_dict = self._create_dummy_data_dict_for_non_standard_input() 
                self.truth_da = self.da_dict[self.truth_data_name]

            print("**** Loading DataArrays")    
            for data_name, da in self.da_dict.items():
                print(f"Loading {data_name} data") 
                da.load()

    ##### PRIVATE METHODS #####
    ############################################################################
    # Check that time dimension is 'period_end_time'
    # Check that south->north dimension is 'lat', 'latitude', 'y', or 'south_north'
    # Check that west->east dimension is 'lon', 'longitude', 'x', or 'west_east'
    def _check_for_standard_dimensions(self, da):
        if (len(da.dims) != 3):
            print(f"Error: Must have three-dimensional grids with dimensions (time, south->north, west->east); current dims are {da.dims}")
            sys.exit(1)

        # Check time dimension
        if (da.dims[0] != utils.period_end_time_dim_str):
            print(f"Error: Only time dimension name {utils.period_end_time_dim_str} supported, not {da.dims[0]}")
            sys.exit(1)
        
        # Check south->north (lat) dimension
        #if (da.dims[1] != "lat"):
        #    print(f"Error: Only south->north dimension names 'lat' currently supported, not {da.dims[1]}")
        #    sys.exit(1)
        if (da.dims[1] != "lat") and \
           (da.dims[1] != "latitude") and \
           (da.dims[1] != "y") and \
           (da.dims[1] != "south_north"):
            print(f"Error: Only south->north dimension names 'lat', 'latitude', 'y', or 'south_north' supported, not {self.truth_da.dims[1]}")
            sys.exit(1)

        # Check west->east (lon) dimension
        #if (da.dims[2] != "lon"):
        #    print(f"Error: Only east->west dimension names 'lon' currently supported, not {self.truth_da.dims[2]}")
        #    sys.exit(1)
        if (da.dims[2] != "lon") and \
           (da.dims[2] != "longitude") and \
           (da.dims[2] != "x") and \
           (da.dims[2] != "west_east"):
            print(f"Error: Only west->east dimension names 'lon', 'longitude', 'x', or 'west_east' supported, not {self.truth_da.dims[2]}")
            sys.exit(1)
       
        if not(self.USE_EXTERNAL_DA_DICT): 
            if not(hasattr(self, "lat_dim_name")):
                self.lat_dim_name = da.dims[1] 
                self.lon_dim_name = da.dims[2]
                self.dims = (self.time_dim_name, self.lat_dim_name, self.lon_dim_name)
            else:
                if (da.dims[1] != self.lat_dim_name):
                    print(f"Error: DataArray south->north dimension name {da.dims[1]} does not match previously set dimension name {self.lat_dim_name}")
                    sys.exit(1)
                if (da.dims[2] != self.lon_dim_name):
                    print(f"Error: DataArray west->east dimension name {da.dims[2]} does not match previously set dimension name {self.lon_dim_name}")
                    sys.exit(1)

    def _create_dummy_data_dict_for_non_standard_input(self):
        da_dict = {}
        da = xr.DataArray(np.arange(10))
        pdp.add_attributes_to_data_array(da, short_name = f"{self.temporal_res}-hour precipitation", units = "mm")
        da_dict[self.truth_data_name] = da

        return da_dict

    def _construct_path_to_input_precip_data(self, data_name):
        if (type(self.temporal_res) is int):
            temporal_res_str = f"{self.temporal_res:02d}_hour_precipitation"
        else:
            temporal_res_str = f"{self.temporal_res}_res_precipitation"

        if (data_name == self.data_grid):
            data_dir = f"{data_name}.{self.native_grid_name}.{temporal_res_str}"
        else:
            data_dir = f"{data_name}.{self.data_grid_name}.{temporal_res_str}"

        if (self.input_format == "zarr"):
            data_dir += ".zarr"

        full_data_dir = os.path.join(self.input_dir, data_dir)

        print(f"Input data directory: {full_data_dir}")
        if (not os.path.exists(full_data_dir)):
            print(f"Error: Input data directory {full_data_dir} does not exist")
            sys.exit(1)

        return full_data_dir, temporal_res_str

    def _construct_valid_dt_lists(self):
        self.start_dt = pdp.check_model_valid_dt_format(self.start_dt_str, resolution = self.temporal_res) 
        self.end_dt = pdp.check_model_valid_dt_format(self.end_dt_str, resolution = self.temporal_res)
        
        # Construct annual datetime list (for annual stats, timeseries, etc.)
        # Because data we're working with are period-ending, if self.end_dt is, for example, 20090101.00,
        # we don't actually have data over the final year (2009 in this example). Remove final year from self.valid_annual_dt_list. 
        self.first_year = dt.datetime.strptime(self.start_dt_str[:4], "%Y")
        self.final_year = dt.datetime.strptime(self.end_dt_str[:4], "%Y")
        self.valid_annual_dt_list = list(dateutil.rrule.rrule(dateutil.rrule.YEARLY, dtstart = self.first_year, until = self.final_year)) 
        if (self.end_dt.month == 1) and (self.end_dt.day == 1) and (self.end_dt.hour == 0):
            self.valid_annual_dt_list = self.valid_annual_dt_list[:-1]
        self.annual_time_period_str = f"{self.valid_annual_dt_list[0]:%Y}-{self.valid_annual_dt_list[-1]:%Y}"
        
        # Construct monthly datetime list (for monthly stats, timeseries, etc.)
        # Because data we're working with are period-ending, if self.end_dt is, for example, 20080601.00,
        # we don't actually have data over the final month (June 2008 in this example). Remove last month from self.valid_monthly_dt_list. 
        self.first_month = dt.datetime.strptime(self.start_dt_str[:6], "%Y%m")
        self.final_month = dt.datetime.strptime(self.end_dt_str[:6], "%Y%m")
        self.valid_monthly_dt_list = list(dateutil.rrule.rrule(dateutil.rrule.MONTHLY, dtstart = self.first_month, until = self.final_month))
        if (self.end_dt.day == 1) and (self.end_dt.hour == 0):
            self.valid_monthly_dt_list = self.valid_monthly_dt_list[:-1]
        self.monthly_time_period_str = f"{self.valid_monthly_dt_list[0]:%Y%m}-{self.valid_monthly_dt_list[-1]:%Y%m}"

        # Construct daily datetime list (which will correspond to cadence of input netCDF files)
        current_daily_dt = dt.datetime(self.start_dt.year, self.start_dt.month, self.start_dt.day)
        end_daily_dt = dt.datetime(self.end_dt.year, self.end_dt.month, self.end_dt.day)
        self.valid_daily_dt_list = [current_daily_dt]
        while (current_daily_dt != end_daily_dt):
            current_daily_dt += dt.timedelta(days = 1)
            self.valid_daily_dt_list.append(current_daily_dt)
        if (len(self.valid_daily_dt_list) > 1):
            self.valid_daily_dt_list_period_begin = [dtime - dt.timedelta(days = 1) for dtime in self.valid_daily_dt_list[1:]]
        else:
            self.valid_daily_dt_list_period_begin = self.valid_daily_dt_list 
        self.daily_time_period_str = f"{self.valid_daily_dt_list[0]:%Y%m%d}-{self.valid_daily_dt_list[-1]:%Y%m%d}"
        self.daily_time_period_str_period_begin = f"{self.valid_daily_dt_list_period_begin[0]:%Y%m%d}-{self.valid_daily_dt_list_period_begin[-1]:%Y%m%d}"

        # Construct key datetime list self.valid_dt_list, which corresponds to the temporal
        # resolution of the data and represents period-END times of the data (e.g., the first
        # valid time for 24-hour precip is the beginning of the second day). 
        if (self.temporal_res == 24):
            self.valid_dt_list = self.valid_daily_dt_list[1:]
        elif (self.temporal_res == 12) or \
             (self.temporal_res ==  6) or \
             (self.temporal_res ==  3) or \
             (self.temporal_res ==  1):
            current_dt = self.start_dt
            self.valid_dt_list = [current_dt]
            while (current_dt != self.end_dt):
                current_dt += dt.timedelta(hours = self.temporal_res)
                self.valid_dt_list.append(current_dt)
        else:
            print(f"Verification for {self.temporal_res} temporal resolution not yet implemented")
            sys.exit(0) 
        self.time_period_str = f"{self.valid_dt_list[0]:%Y%m%d.%H}-{self.valid_dt_list[-1]:%Y%m%d.%H}"
        self.num_valid_times = len(self.valid_dt_list)
        self.first_valid_dtime = self.valid_dt_list[0]
        self.final_valid_dtime = self.valid_dt_list[-1]
        self.first_valid_dtime_str = self.first_valid_dtime.strftime(utils.full_date_format_str) 
        self.final_valid_dtime_str = self.final_valid_dtime.strftime(utils.full_date_format_str)

        print(f"Number of valid times: {self.num_valid_times}")
        print(f"Annual time period: {self.annual_time_period_str}")
        print(f"Monthly time period: {self.monthly_time_period_str}")
        print(f"Daily time period: {self.daily_time_period_str}")
        print(f"Daily time period (period-beginning): {self.daily_time_period_str_period_begin}")
        print(f"Data resolution time period: {self.time_period_str}")

    def _create_region_mask(self, data_array):
        if (self.region == "US-Mountain"):
            self.region_mask = ppu.create_mountain_states_mask(data_array)
        elif (self.region == "US-WestCoast"):
            self.region_mask = ppu.create_west_coast_states_mask(data_array) 
        elif ("US" in self.region):
            self.region_mask = ppu.create_conus_mask(data_array)
        elif (self.region == "Africa") or \
             (self.region == "Australia") or \
             (self.region == "Europe") or \
             (self.region == "SouthAmerica"):
            self.region_mask = ppu.create_continent_mask(data_array, self.region)
        else:
            self.region_mask = None
            return data_array

        # Use the xarray.DataArray.where method to drop any values (i.e., turn them into nans) that don't fall within the region mask
        data_array = data_array.where(self.region_mask)
        return data_array 

    def _read_region_subset_and_load_data(self):
        self.da_dict = {} # DataArray dictionary containing region-subsetted data
        if self.LOAD_DATA:
            self.loaded_non_subset_da_dict = {} # DataArray dictionary containing data as it was read in (not region-subsetted)
            for data_name in self.data_names:
                print(f"**** Reading dataset {data_name}")
                dataset_dir, temporal_res_str = self._construct_path_to_input_precip_data(data_name)

                if (self.input_format == "netcdf"):
                    # Collect netCDF file list
                    file_list = []
                    for dtime in self.valid_daily_dt_list:
                        if (data_name == self.data_grid):
                            fname = f"{data_name}.{self.native_grid_name}.{temporal_res_str}.{dtime:%Y%m%d}.nc"
                        else:
                            fname = f"{data_name}.{self.data_grid_name}.{temporal_res_str}.{dtime:%Y%m%d}.nc"

                        fpath = os.path.join(dataset_dir, fname)
                        if not(os.path.exists(fpath)):
                            print(f"Warning: Input file path {fpath} does not exist; not including in input file list")
                            continue
                        file_list.append(fpath)

                    if (len(file_list) == 0):
                        print(f"Error: No input files found in directory {dataset_dir}; can't proceed with verification")
                        sys.exit(1)

                    # Read multi-file dataset
                    dataset = xr.open_mfdataset(file_list)
                elif (self.input_format == "zarr"):
                    dataset = xr.open_zarr(dataset_dir)

                precip_da = dataset[f"precipitation_{self.temporal_res:02d}_hour"]
                self._check_for_standard_dimensions(precip_da)
                precip_da.attrs["data_name"] = data_name

                # Index obs data array to correct datetime range
                precip_da = precip_da.loc[self.first_valid_dtime_str:self.final_valid_dtime_str]
                # Old method: doesn't work for zarr datasets which contain full range of data (end up with one time step too many)
                #precip_da = precip_da.loc[self.start_dt.strftime(utils.full_date_format_str):self.end_dt.strftime(utils.full_date_format_str)]
        
                # Add interval_hours attribute to DataArrays, which is not included in some datasets like IMERG
                pdp.add_attributes_to_data_array(precip_da, interval_hours = self.temporal_res)

                # Retain copies of non-region-subsetted data arrays (in case we want to subsequently subset to other regions)
                self.loaded_non_subset_da_dict[data_name] = precip_da

        # Subset data to region
        for data_name, precip_da in self.loaded_non_subset_da_dict.items():
            self.da_dict[data_name] = self._subset_data_to_region(precip_da.copy(), data_name = data_name)

        # Ensure the shapes of all the data arrays are the same. 
        print("**** Checking consistency of DataArray shapes")    
        self.truth_da = self.da_dict[self.truth_data_name] 
        truth_da_shape = self.truth_da.shape
        for data_name, da in self.da_dict.items():
            if (da.shape != truth_da_shape): 
                print(f"Error: {data_name} data {da.shape} has a different shape than {self.truth_data_name} {truth_da_shape}; not proceeding with verification")
                sys.exit(1)
            print(f"{data_name} data {da.shape} has the same shape as {self.truth_data_name} data {truth_da_shape}")

        # Load each DataArray. Maintain a separate loop for this in case any of the DataArrays are
        # found to be the wrong shape above. We don't want to spend time loading each DataArray
        # only to discover that one is the wrong shape and we can't perform verification.
        print("**** Loading DataArrays")    
        for data_name, da in self.da_dict.items():
            print(f"Loading region-subsetted {data_name} data") 
            da.load()
   
    # If truth_data_name is not the first index, swap it with whatever
    # is the first index 
    def _reorder_data_names_with_truth_data_first(self):
        truth_data_index = self.data_names.index(self.truth_data_name)
        if (truth_data_index != 0):
            self.data_names[truth_data_index] = self.data_names[0]
            self.data_names[0] = self.truth_data_name 

    # Set format of input datasets (netcdf, zarr, etc.)
    def _set_input_format(self, input_format, name_input_dir_by_input_format = True):
        if (input_format == "netcdf") or (input_format == "nc"):
            self.input_format = "netcdf"
            if name_input_dir_by_input_format:
                self.input_dir = os.path.join(self.input_dir, "netcdf") 
        elif (input_format == "zarr"):
            self.input_format = "zarr"
            if name_input_dir_by_input_format:
                self.input_dir = os.path.join(self.input_dir, "zarr") 
        else:
            print(f"Error: Unsupported input format {self.input_format}")
            sys.exit(1) 

    def _set_region_info(self, region, region_info):
        self.region = region
        
        # If data are on the Replay grid, longitudes go from 0 to 360,
        # rather than -180  to 180, and the latitude coordinates are flipped,
        # i.e., they go from +90 to -90 (north to south).
        # FIXME: Generalize how this flag is set rather than hard-coding the grid names that have 0->360 longitudes.
        self.LONS_360_FLAG = False 
        self.LATS_FLIP_FLAG = False
        if (self.data_grid == "Replay"):
            self.LONS_360_FLAG = True 
            self.LATS_FLIP_FLAG = True 
        elif (self.data_grid == "NestedEagle"):
            self.LONS_360_FLAG = True 
            self.LATS_FLIP_FLAG = False

        if (region_info is None):
            if self.region in ppu.regions_info_dict.keys():
                self.region_plot_config = ppu.regions_info_dict[self.region]  
            else:
                print(f"Error: No region info provided for region {region}, and this region is not configured in ppu.regions_info_dict")
                sys.exit(1)
        else:
            self.region_plot_config = ppu.RegionPlottingConfiguration(region_extent = region_info["region_extent"], 
                                                                      figsize_sp = region_info["figsize_sp"], 
                                                                      figsize = region_info["figsize"],
                                                                      figsize_errors = region_info["figsize_errors"],
                                                                      subplot_layout = region_info["subplot_layout"],
                                                                      cm_mean_precip_range = region_info["cm_mean_precip_range"],
                                                                      ts_mean_precip_range = region_info["ts_mean_precip_range"],
                                                                      central_point = region_info["central_point"],
                                                                      crosses_meridian = region_info["crosses_meridian"])
        self.region_extent = self.region_plot_config.region_extent
        self.region_spans_meridian = self.region_plot_config.crosses_meridian

        # Check that region_extent is interpretable
        if (type(self.region_extent) is not list) or \
            (len(self.region_extent) != 4) or \
            not(isinstance(self.region_extent[0], (int, float))) or \
            not(isinstance(self.region_extent[1], (int, float))) or \
            not(isinstance(self.region_extent[2], (int, float))) or \
            not(isinstance(self.region_extent[3], (int, float))) or \
            (np.abs(self.region_extent[0]) > 180) or \
            (np.abs(self.region_extent[1]) > 180) or \
            (np.abs(self.region_extent[2]) > 90)  or \
            (np.abs(self.region_extent[3]) > 90):
            print(f"Error: Region extent info {self.region_extent} is non-standard and cannot be interpreted")
            sys.exit(1)

        # Index by lat/lon corresponding to current region. To do this,
        # recall data goes from [0, 360) longitude and (90, -90) latitude.
        # So the longitude bounds must be adjusted first
        self.lower_lon = self.region_extent[0]
        self.upper_lon = self.region_extent[1]
        self.lower_lat = self.region_extent[2]
        self.upper_lat = self.region_extent[3]
        if (self.LONS_360_FLAG):
            if (self.lower_lon < 0):
                self.lower_lon += 360.0
            if (self.upper_lon < 0):
                self.upper_lon += 360.0

    # FIXME: Discontinuity in IMERG and ERA5 data in Europe and Africa where data crosses meridian 
    def _subset_data_to_region(self, data_array, data_name = None):
        print(f"Subsetting {data_name} data to region {self.region}")

        if (self.region == "Global"):
            return data_array
        else:
            # For 2D lat/lon (e.g., curvilinear) grids, mask the data using a mask set to within the bounds of the
            # region's lat/lons.
            # Potential TODO: This methodology may work for 1D rectilinear grids, too (may not need the else-statement with the slicing below)
            if self.LAT_LON_2D:
               mask = ((data_array.lat >= self.lower_lat) & (data_array.lat <= self.upper_lat) & \
                      (data_array.lon >= self.lower_lon) & (data_array.lon <= self.upper_lon)).compute()
               region_subset_data_array = data_array.where(mask, drop = True) 
            else:
                if (self.region_spans_meridian) and (self.LONS_360_FLAG):
                    if (self.LATS_FLIP_FLAG):
                        region_subset_west_of_meridian = data_array.sel(lat = slice(self.upper_lat, self.lower_lat), lon = slice(self.lower_lon, 360.0)) 
                        region_subset_east_of_meridian = data_array.sel(lat = slice(self.upper_lat, self.lower_lat), lon = slice(0.0, self.upper_lon))
                    else:
                        region_subset_west_of_meridian = data_array.sel(lat = slice(self.lower_lat, self.upper_lat), lon = slice(self.lower_lon, 360.0)) 
                        region_subset_east_of_meridian = data_array.sel(lat = slice(self.lower_lat, self.upper_lat), lon = slice(0.0, self.upper_lon))
                    region_subset_data_array = xr.concat([region_subset_west_of_meridian, region_subset_east_of_meridian], dim = self.lon_dim_name) 

                    # The slicing above will leave us with longitudes that go from (for example), ~340 to 360, then start over at zero. In other words,
                    # they are numerically out of order, which makes further manipulation and plotting of the array difficult.
                    # So, modify the longitudes to be increasing from negative (west of meridian) to positive (east of meridian),
                    # then reorder the data array by longitude accordingly.
                    lons_m180to180 = utils.longitude_to_m180to180(region_subset_data_array[self.lon_dim_name].values)
                    region_subset_data_array[self.lon_dim_name] = lons_m180to180
                    region_subset_data_array = region_subset_data_array.sortby(self.lon_dim_name) 
                else:
                    if (self.LATS_FLIP_FLAG):
                        region_subset_data_array = data_array.sel(lat = slice(self.upper_lat, self.lower_lat), lon = slice(self.lower_lon, self.upper_lon))
                    else:
                        region_subset_data_array = data_array.sel(lat = slice(self.lower_lat, self.upper_lat), lon = slice(self.lower_lon, self.upper_lon))

        print(f"Region subset data array shape: {region_subset_data_array.shape}")
        # For certain regions, take an extra step and mask the data to only the land/geopolitical boundaries of that region.
        # For CONUS, for example, avoids including grid points from the ocean, Mexico, and Canada in the stats (e.g., the very high precip Gulf Stream area).
        print(f"Masking data to only the geopolitical boundaries of this region (if applicable)")
        region_subset_data_array = self._create_region_mask(region_subset_data_array)

        print(f"{data_name} data array shape, subsetted to region: {region_subset_data_array.shape}")
        return region_subset_data_array 

    ##### PUBLIC METHODS #####
    ############################################################################
    ##### Getter methods #####
    def get_num_valid_times(self) -> int: 
        return self.num_valid_times

    def get_summed_data(self) -> Dict[str, Any]:
        if not(hasattr(self, "da_dict_summed_time")):
            self.sum_data_over_full_time_period()
 
        return self.da_dict_summed_time

    ##### Public methods stats calculations #####
    # Calculate occurrence statistics over entire forecast and obs datasets (i.e., the stats will be derived
    # from grids that are valid in time and space).
    def calculate_occ_stats(self, input_da_dict = None,
                            threshold_list = utils.default_eval_threshold_list_mm): 
        if (input_da_dict is None):
            input_da_dict = self.da_dict

        self.threshold_da_for_occ_stats = xr.DataArray(threshold_list)
        pdp.add_attributes_to_data_array(self.threshold_da_for_occ_stats, units = "mm")
 
        occ_stats_dict = {}
        obs_precip = input_da_dict[self.truth_data_name]
        for data_name, da in input_da_dict.items():
            if (data_name == self.truth_data_name):
                continue
            model_precip = input_da_dict[data_name] 
            print(f"Calculating occurence statistics for dataset {data_name}")

            # Calculate FSS for varying evaluation radius, fixed threshold
            hits_list = []
            misses_list = []
            false_alarms_list = []
            correct_negatives_list = []
            total_events_list = []
            frequency_bias_list = []
            CSI_list = []
            ETS_list = []
            for threshold in threshold_list:
                hits = self.calculate_hits(threshold, model_precip, obs_precip)
                hits_list.append(hits)

                misses = self.calculate_misses(threshold, model_precip, obs_precip)
                misses_list.append(misses)
                
                false_alarms = self.calculate_false_alarms(threshold, model_precip, obs_precip)
                false_alarms_list.append(false_alarms)

                correct_negatives = self.calculate_correct_negatives(threshold, model_precip, obs_precip)
                correct_negatives_list.append(correct_negatives)

                total_events = hits + misses + false_alarms + correct_negatives
                total_events_list.append(total_events)

                # Frequency bias
                # Measures the ratio of the frequency of forecast events to the frequency of observed events
                # See https://www.cawcr.gov.au/projects/verification/verif_web_page.html#Methods_for_dichotomous_forecasts
                if (hits + misses > 0):
                    frequency_bias = (hits + false_alarms)/(hits + misses)
                else:
                    frequency_bias = np.nan 
                frequency_bias_list.append(frequency_bias)

                # CSI (Critical Success Index) AKA TS (Threat Score)
                # Measures the fraction of observed and/or forecast events that were correctly predicted
                # See https://www.cawcr.gov.au/projects/verification/verif_web_page.html#Methods_for_dichotomous_forecasts
                if (hits + misses + false_alarms > 0):
                    CSI = hits/(hits + misses + false_alarms)
                else:
                    CSI = np.nan
                CSI_list.append(CSI)

                # ETS (Equitable Threat Score) AKA Gilbert Skill Score
                # Measures the fraction of observed and/or forecast events that were correctly predicted, adjusted for hits associated with random chance
                # https://www.cawcr.gov.au/projects/verification/verif_web_page.html#Methods_for_dichotomous_forecasts
                hits_random = (hits + misses) * (hits + false_alarms) / total_events
                if (hits + misses + false_alarms - hits_random > 0): 
                    ETS = (hits - hits_random)/(hits + misses + false_alarms - hits_random)
                else:
                    ETS = np.nan
                ETS_list.append(ETS)
    
            hits_da = self._convert_occ_stats_np_array_to_data_array(np.array(hits_list), self.threshold_da_for_occ_stats, "hits")
            misses_da = self._convert_occ_stats_np_array_to_data_array(np.array(misses_list), self.threshold_da_for_occ_stats, "misses")   
            false_alarms_da = self._convert_occ_stats_np_array_to_data_array(np.array(false_alarms_list), self.threshold_da_for_occ_stats, "false_alarms")
            correct_negatives_da = self._convert_occ_stats_np_array_to_data_array(np.array(correct_negatives_list), self.threshold_da_for_occ_stats, "correct_negatives")
            total_events_da = self._convert_occ_stats_np_array_to_data_array(np.array(total_events_list), self.threshold_da_for_occ_stats, "total_events")
            frequency_bias_da = self._convert_occ_stats_np_array_to_data_array(np.array(frequency_bias_list), self.threshold_da_for_occ_stats, "frequency_bias")
            CSI_da = self._convert_occ_stats_np_array_to_data_array(np.array(CSI_list), self.threshold_da_for_occ_stats, "CSI")
            ETS_da = self._convert_occ_stats_np_array_to_data_array(np.array(ETS_list), self.threshold_da_for_occ_stats, "ETS")

            occ_stats_dict[data_name] = StatsDataClass(
                                                       threshold = self.threshold_da_for_occ_stats, 
                                                       hits = hits_da,
                                                       misses = misses_da,
                                                       false_alarms = false_alarms_da,
                                                       correct_negatives = correct_negatives_da,
                                                       total_events = total_events_da,
                                                       frequency_bias = frequency_bias_da,
                                                       CSI = CSI_da,
                                                       ETS = ETS_da
                                                      )

        return occ_stats_dict 

    def extract_occ_stat_dict(self, occ_stats_dict, which_stat):
        stat_dict = {}
        for data_name, stats_data in occ_stats_dict.items():
            match which_stat:
                case "hits":
                    stat_dict[data_name] = stats_data.hits
                case "misses":
                    stat_dict[data_name] = stats_data.misses
                case "false_alarms":
                    stat_dict[data_name] = stats_data.false_alarms
                case "correct_negatives":
                    stat_dict[data_name] = stats_data.correct_negatives
                case "total_events":
                    stat_dict[data_name] = stats_data.total_events
                case "frequency_bias":
                    stat_dict[data_name] = stats_data.frequency_bias
                case "CSI":
                    stat_dict[data_name] = stats_data.CSI
                case "ETS":
                    stat_dict[data_name] = stats_data.ETS
                case _:
                    print(f"Error: Unrecognized occurence stat type {which_stat}")
                    return

        return stat_dict

    def extract_occ_stat_data_array(self, occ_stats_dict, which_stat, data_name):
        match which_stat:
            case "hits":
                return occ_stats_dict[data_name].hits 
            case "misses":
                return occ_stats_dict[data_name].misses 
            case "false_alarms":
                return occ_stats_dict[data_name].false_alarms 
            case "correct_negatives":
                return occ_stats_dict[data_name].correct_negatives 
            case "total_events":
                return occ_stats_dict[data_name].total_events 
            case "frequency_bias":
                return occ_stats_dict[data_name].frequency_bias 
            case "CSI":
                return occ_stats_dict[data_name].CSI 
            case "ETS":
                return occ_stats_dict[data_name].ETS 
            case _:
                print(f"Error: Unrecognized occurence stat type {which_stat}")
                return

    # Calculate correlation coefficient
    def calculate_pearsonr(self, model_precip, obs_precip):
        model_precip_values_flat = model_precip.values.flatten()
        obs_precip_values_flat = obs_precip.values.flatten()
        model_precip_no_nans = model_precip_values_flat[~np.isnan(model_precip_values_flat) & ~np.isnan(obs_precip_values_flat)]
        obs_precip_no_nans = obs_precip_values_flat[~np.isnan(model_precip_values_flat) & ~np.isnan(obs_precip_values_flat)]
        pearsonr = scipy.stats.pearsonr(model_precip_no_nans, obs_precip_no_nans)
        return pearsonr

    # Calculate RMSE
    def calculate_rmse(self, model_precip, obs_precip):
        squared_errors = (model_precip - obs_precip)**2
        return np.sqrt(squared_errors.mean().load()).item()

    # Calculate mean amount bias
    def calculate_bias(self, model_precip, obs_precip):
        return (model_precip - obs_precip).mean().item()
    
    # Calculate hits
    def calculate_hits(self, threshold, model_precip, obs_precip):
        hits_condition = (model_precip >= threshold) & (obs_precip >= threshold)
        number_of_hits = len(np.where(hits_condition.values.flatten())[0])
        return number_of_hits

    # Calculate misses
    def calculate_misses(self, threshold, model_precip, obs_precip):
        misses_condition = (model_precip < threshold) & (obs_precip >= threshold)
        number_of_misses = len(np.where(misses_condition.values.flatten())[0]) 
        return number_of_misses

    # Calculate false alarms
    def calculate_false_alarms(self, threshold, model_precip, obs_precip):
        false_alarms_condition = (model_precip >= threshold) & (obs_precip < threshold)
        number_of_false_alarms = len(np.where(false_alarms_condition.values.flatten())[0]) 
        return number_of_false_alarms

    # Calculate correct negatives
    def calculate_correct_negatives(self, threshold, model_precip, obs_precip):
        correct_negatives_condition = (model_precip < threshold) & (obs_precip < threshold)
        number_of_correct_negatives = len(np.where(correct_negatives_condition.values.flatten())[0])
        return number_of_correct_negatives

    def how_to_calculate_aggregated_stats(self):
        print('calculate_aggregated_stats(input_da_dict = None, time_period_type = None\n'
              '                           agg_type = "space_time", stat_type = "mean",\n'
              '                           pctl = 99, include_zeros = True, write_to_nc = False)\n')
        print("NOTE:\n"
              "If input_da_dict is None, self.da_dict will be aggregated according to time_period_type, agg_type, and stat_type.")

    # Calculate statistics valid for data aggregated in space, time, or space and time. Currently
    # only mean and percentile stats are supported. 
    def calculate_aggregated_stats(self,
                                   input_da_dict = None,
                                   time_period_type = None, 
                                   agg_type = "space_time",         
                                   stat_type = "mean",
                                   pctl = 99,
                                   write_to_nc = False):
        if ("pctl" in stat_type): 
            print(f"Calculating {agg_type}-aggregated {time_period_type} {pctl:0.1f}th {stat_type}")
        else:
            print(f"Calculating {agg_type}-aggregated {time_period_type} {stat_type}")

        if input_da_dict is None:
            input_da_dict = self.da_dict
        
        # Process time_period_type: list of date times, dimension name, etc.
        dtimes, dim_name, time_period_str = self._process_time_period_type_to_dtimes(time_period_type)

        # Process agg_type: determine which dimension(s) to aggregate over
        match agg_type:
            case "space": # Is this even needed? It would be (for example), a spatial mean at each valid time
                agg_dims = (self.lat_dim_name, self.lon_dim_name)
            case "time":
                agg_dims = (utils.period_begin_time_dim_str)
            case "space_time":
                agg_dims = (utils.period_begin_time_dim_str, self.lat_dim_name, self.lon_dim_name) 

        # Process stat_type: eventual attributes of aggregated data arrays
        short_name = self.truth_da.short_name
        long_name = self.truth_da.long_name
        time_period_type_str = ""
        if (time_period_type is not None):
            time_period_type_str = f"{time_period_type} "
        match stat_type:
            case "mean":
                short_name = short_name + f" {time_period_type_str}{stat_type}"
                long_name = f"{time_period_type_str.title()}{stat_type} of " + long_name
            case "mean_exclude_zeros": 
                short_name = short_name + f" {time_period_type_str}{stat_type}"
                long_name = f"{time_period_type_str.title()}{stat_type} of " + long_name
            case "max":
                short_name = short_name + f" {time_period_type_str}{stat_type}"
                long_name = f"{time_period_type_str.title()}{stat_type} of " + long_name 
            case "pctl":
                short_name = short_name + f" {time_period_type_str}{pctl:0.1f}th {stat_type}"
                long_name = f"{pctl:0.1f}th {stat_type} of " + long_name
            case "pctl_exclude_zeros":
                short_name = short_name + f" {time_period_type_str}{pctl:0.1f}th {stat_type}"
                long_name = f"{pctl:0.1f}th {stat_type} of " + long_name
            case _:
                print(f"Error: Unrecognized stat type {stat_type}")
                return

        agg_data_dict = {}
        for data_name, da in input_da_dict.items():
            # Convert data coordinates to period beginning (easier to aggregate over different time periods this way)
            if (time_period_type is not None):
                data_array = utils.convert_period_end_to_period_begin(da)
            else:
                data_array = da

            # If aggregating seasonally, dtimes, which in this case will be a list of lists defining
            # seasonal datetime ranges, wasn't defined above.
            if (time_period_type == "seasonal"):
                dtimes = self._construct_season_dt_ranges(data_array)

            # Calculate spatiotemporal means across regions and months
            data_list = [] 
            for dtime in dtimes:
                data_to_aggregate = self._determine_agg_data_from_time_period_type(data_array, time_period_type, dtime)

                match stat_type:
                    case "mean":
                        data = data_to_aggregate.mean(dim = agg_dims)
                    case "mean_exclude_zeros":
                        data = data_to_aggregate.where(data_to_aggregate > 0.0).mean(dim = agg_dims)
                    case "max":
                        data = data_to_aggregate.max(dim = agg_dims)
                    case "pctl":
                        data = data_to_aggregate.quantile(pctl/100, dim = agg_dims)
                    case "pctl_exclude_zeros":
                        data = data_to_aggregate.where(data_to_aggregate > 0.0).quantile(pctl/100, dim = agg_dims)
                data_list.append(data)

            # Convert data to xarray DataArray via xr.concat
            agg_da = xr.concat(data_list, dim = dim_name)
            if (time_period_type == "seasonal"):
                agg_da.coords[dim_name] = self._construct_seasonal_dt_str_list(dtimes) 
            else:
                agg_da.coords[dim_name] = dtimes 
            pdp.add_attributes_to_data_array(agg_da,
                                             short_name = short_name, 
                                             long_name = long_name, 
                                             units = da.units)
            agg_data_dict[data_name] = agg_da 

            # Output to netCDF
            if write_to_nc:
                match stat_type:
                    case "mean":
                        stat_type_out_str = stat_type
                    case "mean_exclude_zeros":
                        stat_type_out_str = stat_type
                    case "max":
                        stat_type_out_str = stat_type
                    case "pctl":
                        stat_type_out_str = f"{pctl:0.1f}th_{stat_type}"
                    case "pctl_exclude_zeros":
                        stat_type_out_str = f"{pctl:0.1f}th_{stat_type}"
                    case _:
                        stat_type_out_str = stat_type

                self._set_output_var_name(agg_da) 
                nc_out_fpath = self._configure_output_stats_nc_fpath(data_name, time_period_str, time_period_type = time_period_type,
                                                                     stat_type = stat_type_out_str, agg_type = agg_type)
                print(f"Writing {nc_out_fpath}")
                agg_da.to_netcdf(nc_out_fpath)

        return agg_data_dict

    def calculate_points_above_threshold(self, input_da_dict = None, threshold = 1.0, gte = True):
        if (input_da_dict is None):
            input_da_dict = self.da_dict

        dtimes, time_dim, dt_format = self._create_datetime_list_from_da_time_dim(self.truth_da)

        above_thresh_dict = {}
        for data_name, da in input_da_dict.items():
            points_above_threshold_list = []
            for d in range(len(da.period_end_time.values)):
                if gte:
                    gte_string = "gte"
                    long_name_prefix = "Number of points greater than or equal to"
                    points_above_threshold = np.where(da.values[d,:,:].flatten() >= threshold)[0].shape[0]
                else:
                    points_above_threshold = np.where(da.values[d,:,:].flatten() > threshold)[0].shape[0]
                    gte_string = "gt"
                    long_name_prefix = "Number of points greater than"
                points_above_threshold_list.append(points_above_threshold)

            above_thresh_da = xr.DataArray(points_above_threshold_list, dims = ["period_end_time"], coords = [dtimes])
            pdp.add_attributes_to_data_array(above_thresh_da,
                                             short_name = f"Points {gte_string} {threshold:0.2f}mm",
                                             long_name = f"{long_name_prefix} {threshold:0.2f} mm",
                                             units = "Num points")
            above_thresh_da.attrs["threshold"] = threshold
            above_thresh_dict[data_name] = above_thresh_da 

        return above_thresh_dict

    # Calculate probability density functions (PDFs)
    def calculate_pdf(self, input_da_dict = None, time_period_type = "full_period", write_to_nc = False, bins = 10):
        if input_da_dict is None:
            input_da_dict = self.da_dict

        # Process time_period_type: list of date times, dimension name, etc.
        dtimes, dim_name, time_period_str = self._process_time_period_type_to_dtimes(time_period_type)

        # If aggregating seasonally, dtimes, which in this case will be a list of lists defining
        # seasonal datetime ranges, wasn't defined above.
        if (time_period_type == "seasonal"):
            dtimes = self._construct_season_dt_ranges(data_array)

        pdf_data_dict = {}
        for dtime in dtimes:
            pdf_each_dtime_dict = {}
            for data_name, da in input_da_dict.items():
                # Convert data coordinates to period beginning (much easier to aggregate over months that way)
                data_array = utils.convert_period_end_to_period_begin(da)

                # Calculate pdf 
                data_to_aggregate = self._determine_agg_data_from_time_period_type(data_array, time_period_type, dtime)

                hist_and_bins = data_to_aggregate.plot.hist(bins = bins)
                hist = hist_and_bins[0]
                bins = hist_and_bins[1]
                total_samples = hist.sum()
                pdf = hist/total_samples

                # Including total samples in order to back out original histogram from PDF
                pdf_each_dtime_dict[data_name] = (pdf, bins, total_samples) 
                    
            pdf_data_dict[dtime] = pdf_each_dtime_dict

        # Output to netCDF
        if write_to_nc:
            num_dtimes = len(dtimes)
            # Loop through each data_name ('AORC', etc.), writing pdf data to a data_name-specific netCDF file
            for data_name in self.data_names:
                # Here, just initialize the _full_array variables as empty arrays with size 0
                bins_full_array = np.empty(0)
                probs_full_array = np.empty(0)
                # Loop through each dtime, concatenating respective pdf and bins data 
                total_samples_list = []
                for dtime in dtimes: 
                    total_samples = pdf_data_dict[dtime][data_name][2]
                    total_samples_list.append(total_samples) 
                    bins = np.expand_dims(pdf_data_dict[dtime][data_name][1], 0)
                    probs = np.expand_dims(pdf_data_dict[dtime][data_name][0], 0)
                    if (bins_full_array.size == 0):
                        bins_full_array = bins
                        probs_full_array = probs
                    else:
                        bins_full_array = np.concatenate( (bins_full_array, bins), axis = 0) 
                        probs_full_array = np.concatenate( (probs_full_array, probs), axis = 0)

                # Convert full arrays of bins and pdf data to DataArrays and subsequently unite in a Dataset 
                bins_dim_coords = np.arange(bins_full_array.shape[1])
                bins_da = xr.DataArray(bins_full_array, dims = [dim_name, "bins_dim"], coords = [dtimes, bins_dim_coords])
                probs_dim_coords = np.arange(probs_full_array.shape[1])
                probs_da = xr.DataArray(probs_full_array, dims = [dim_name, "probs_dim"], coords = [dtimes, probs_dim_coords])
                total_samples_da = xr.DataArray(total_samples_list, dims = [dim_name], coords = [dtimes])
                pdf_ds = xr.Dataset({"bins": bins_da, "probs": probs_da, "total_samples": total_samples_da}) 

                # Output Dataset to netCDF
                nc_out_fpath = self._configure_output_stats_nc_fpath(data_name, time_period_str, time_period_type = time_period_type, stat_type = "pdf")
                print(f"Writing {nc_out_fpath}")
                pdf_ds.to_netcdf(nc_out_fpath)

        return pdf_data_dict 

    def how_to_calculate_fss(self):
        print(f'calculate_fss(da_dict = None,\n'
              f'              eval_type = [{evaluate_by_radius_kw_str}, {evaluate_by_threshold_kw_str}, {evaluate_by_radius_ari_threshold_kw_str}, {evaluate_by_ari_kw_str}],\n'
              f'              fixed_radius = 0.5 [deg],\n'
              f'              fixed_threshold = 10.0 [mm],\n'
              f'              fixed_ari_threshold = 2 [years],\n'
              f'              eval_radius_list = {utils.default_eval_radius_list_deg},\n'
              f'              eval_threshold_list = {utils.default_eval_threshold_list_mm},\n'
              f'              eval_ari_list = {utils.default_eval_ari_list_years},\n'
              f'              time_period_type = "full_period",\n'
              f'              radius_units = "deg",\n'
              f'              is_pctl_threshold = False,\n'
              f'              include_zeros = False,\n'
              f'              write_to_nc = False)')

    # Calculate FSS for all QPF datasets, for all valid times. Output FSS at each
    # valid time to a dictionary of DataArrays, so this dict can subsequently be
    # handled similarly to self.da_dict. The dimensions are (num_valid_times * num_eval_radii [num_thresholds]). 
    def calculate_fss(self,
                      da_dict = None,
                      eval_type = evaluate_by_radius_kw_str, 
                      fixed_radius = 0.5, # in degrees lat/lon
                      fixed_threshold = 10.0, # in mm 
                      fixed_ari_threshold = 2, # in years
                      eval_radius_list = utils.default_eval_radius_list_deg, 
                      eval_threshold_list = utils.default_eval_threshold_list_mm,
                      eval_ari_list = utils.default_eval_ari_list_years,
                      time_period_type = "full_period",
                      radius_units = "deg", # For degrees lat/lon; otherwise km, etc.
                      is_pctl_threshold = False,
                      pctl_over_eval_period = False, # Calculate percentile thresholds over full evaluation period (e.g. all seasons) rather than for each grid
                      include_zeros = False, # Include zeros when calculating percentiles when using percentile thresholds
                      write_to_nc = False):
        if (da_dict is None):
            da_dict = self.da_dict
            ari_duration = self.temporal_res
        else:
            ari_duration = da_dict[self.truth_data_name].interval_hours
        data_names = list(da_dict.keys())

        # Process time_period_type: list of date times, dimension name, etc.
        dtimes, dim_name, time_period_str = self._process_time_period_type_to_dtimes(time_period_type)

        if (is_pctl_threshold) and \
           (pctl_over_eval_period) and \
           (time_period_type != "full_period"):
            print(f"Warning: Calculation of static percentiles over full eval period only implemented for time_period_type = full_period")
            print("Reverting to standard grid-by-grid percentile calculations")
            pctl_over_eval_period = False

        # Initialize string (for figure names) indicating whether percentiles are over full time period
        self.pctl_over_eval_period_str = ""
        if pctl_over_eval_period:
            self.pctl_over_eval_period_str = "_over_eval_period"
        
        if include_zeros:
            pctl_stat_type = "pctl"
        else:
            pctl_stat_type = "pctl_exclude_zeros"

        # Calculate percentiles over full evaluation period, to be used for FSS calculations, using percentile thresholds,
        # if pctl_over_eval_period = True (instead of using percentile thresholds calculated for each individual grid).
        pctl_over_eval_period_dict = {}
        if (eval_type == evaluate_by_radius_kw_str): # Evaluate by radius
            fixed_pctl_dict = self.calculate_aggregated_stats(time_period_type = "full_period", agg_type = "space_time",
                                                              stat_type = pctl_stat_type, pctl = fixed_threshold)
            for data_name in data_names: 
                pctl_over_eval_period_dict[data_name] = fixed_pctl_dict[data_name].item()
        else: # Evaluate by threshold
            for p, pctl_threshold in enumerate(eval_threshold_list):
                current_pctl_dict = self.calculate_aggregated_stats(time_period_type = "full_period", agg_type = "space_time",
                                                                    stat_type = pctl_stat_type, pctl = pctl_threshold)

                for data_name in data_names: 
                    if (p == 0):
                        pctl_over_eval_period_dict[data_name] = [ current_pctl_dict[data_name].item() ]
                    else:
                        pctl_over_eval_period_dict[data_name].append(current_pctl_dict[data_name].item())

            # Convert to DataArrays
            # FIXME: Is there a way to do this without another loop through the pctl_over_eval_period_dict dictionary?
            for data_name, pctl_vals_list in pctl_over_eval_period_dict.items():
                pctl_over_eval_period_dict[data_name] = xr.DataArray(pctl_vals_list, dims = ["pctl"], coords = [eval_threshold_list])

        self.fss_eval_radius_units = radius_units
        if is_pctl_threshold:
            self.fss_eval_threshold_units = "pctl"
        else:
            self.fss_eval_threshold_units = "mm"

        if is_pctl_threshold:
            threshold_units = "th_pctl"
            fss_data_dim_name = "pctl_threshold"
        else:
            threshold_units = "mm"
            fss_data_dim_name = "threshold"

        if (eval_type == evaluate_by_radius_kw_str):
            print(f"**** Calculating FSS by radius (fixed threshold {fixed_threshold}{threshold_units})")
            self.fixed_fss_eval_threshold = fixed_threshold
            
            self.fss_eval_radius_da = xr.DataArray(eval_radius_list)
            pdp.add_attributes_to_data_array(self.fss_eval_radius_da, units = self.fss_eval_radius_units)
            fss_data_coords = self.fss_eval_radius_da 

            fss_data_dim_name = "radius"
            stat_type = f"fss.by_{fss_data_dim_name}.thresh{self.fixed_fss_eval_threshold:0.1f}{threshold_units}"
        elif (eval_type == evaluate_by_radius_ari_threshold_kw_str):
            print(f"**** Calculating FSS by radius, using {fixed_ari_threshold:04d}-year ARI grid as threshold")
            self.fixed_fss_eval_threshold = fixed_ari_threshold

            self.fss_eval_radius_da = xr.DataArray(eval_radius_list)
            pdp.add_attributes_to_data_array(self.fss_eval_radius_da, units = self.fss_eval_radius_units)
            fss_data_coords = self.fss_eval_radius_da 
            
            fss_data_dim_name = "radius"
            stat_type = f"fss.by_{fss_data_dim_name}.thresh{self.fixed_fss_eval_threshold:04d}_year_ari"

            self.fixed_ari_grid = self._open_ari_threshold_grid(self.fixed_fss_eval_threshold, ari_duration)
        elif (eval_type == evaluate_by_ari_kw_str):
            print(f"**** Calculating FSS by ARI grid threshold (fixed eval radius {fixed_radius} {self.fss_eval_radius_units})")
            self.fixed_fss_eval_radius = fixed_radius

            self.fss_eval_ari_da = xr.DataArray(eval_ari_list)
            pdp.add_attributes_to_data_array(self.fss_eval_ari_da, units = "years")
            fss_data_coords = self.fss_eval_ari_da

            fss_data_dim_name = "ARI_threshold"
            stat_type = f"fss.by_{fss_data_dim_name}.radius{self.fixed_fss_eval_radius:0.1f}{self.fss_eval_radius_units}"

            # Read in ARI grids that will be used as thresholds for FSS calculations
            self.ari_grid_dict = {}
            for ari in eval_ari_list: 
                self.ari_grid_dict[ari] = self._open_ari_threshold_grid(ari, ari_duration)
        else: # If anything else is passed for <eval_type>, evaluate against threshold
            if is_pctl_threshold:
                pctl_string = "pctl"
            else:
                pctl_string = "amount (mm)"

            print(f"**** Calculating FSS by {pctl_string} threshold (fixed eval radius {fixed_radius} {self.fss_eval_radius_units})")
            self.fixed_fss_eval_radius = fixed_radius

            self.fss_eval_threshold_da = xr.DataArray(eval_threshold_list)
            pdp.add_attributes_to_data_array(self.fss_eval_threshold_da, units = threshold_units)
            fss_data_coords = self.fss_eval_threshold_da 

            stat_type = f"fss.by_{fss_data_dim_name}.radius{self.fixed_fss_eval_radius:0.1f}{self.fss_eval_radius_units}"

        da_dict_fss = {}
        da_dict_f_model = {}
        truth_da = da_dict[self.truth_data_name]
        valid_dt_list = [pd.Timestamp(i) for i in truth_da.period_end_time.values]
        for data_name, da in da_dict.items():
            print(f"Calculating FSS for dataset {data_name}")
            f_obs_list = []
            f_model_list = []
            for v, valid_dt in enumerate(valid_dt_list):
                valid_dt_str = f"{valid_dt:%Y-%m-%d %H:%M:%S}"
                qpe = truth_da.sel(period_end_time = valid_dt_str)

                # For the observations grid (truth dataset), calculate the F_obs values only 
                # (fractions of observed grid exceeding threshold)
                if (data_name == self.truth_data_name): 
                    if (eval_type == evaluate_by_radius_kw_str):
                        if is_pctl_threshold:
                            if include_zeros:
                                threshold_for_F_obs_calc = qpe.quantile(fixed_threshold/100.0)
                            else:
                                threshold_for_F_obs_calc = qpe.where(qpe > 0.0).quantile(fixed_threshold/100.0)
                        else:
                            threshold_for_F_obs_calc = self.fixed_fss_eval_threshold
                        binary_qpe = self._mask_data_array_based_on_threshold(qpe, threshold_for_F_obs_calc) 
                        F_obs = np.where(binary_qpe)[0].shape[0]/binary_qpe.flatten().shape[0]
                        f_obs_list.append(F_obs)
                    elif (eval_type == evaluate_by_radius_ari_threshold_kw_str):
                        binary_qpe = self._mask_data_array_based_on_threshold(qpe, self.fixed_ari_grid)
                        F_obs = np.where(binary_qpe)[0].shape[0]/binary_qpe.flatten().shape[0]
                        f_obs_list.append(F_obs)
                    continue

                # Calculate FSS
                qpf = da.sel(period_end_time = valid_dt_str)
                fss_list = []
                if (eval_type == evaluate_by_radius_kw_str):
                    for radius in eval_radius_list:
                        FSS = self._calculate_fss_single_grid(qpf, qpe, radius, fixed_threshold,
                                                              is_pctl_threshold = is_pctl_threshold,
                                                              pctl_over_eval_period = pctl_over_eval_period,
                                                              external_pctl_value_qpf = pctl_over_eval_period_dict[data_name],
                                                              external_pctl_value_qpe = pctl_over_eval_period_dict[self.truth_data_name],
                                                              include_zeros = include_zeros)
                        fss_list.append(FSS)
                elif (eval_type == evaluate_by_radius_ari_threshold_kw_str):
                    for radius in eval_radius_list:   
                        FSS = self._calculate_fss_single_grid(qpf, qpe, radius, self.fixed_ari_grid)
                        fss_list.append(FSS)
                elif (eval_type == evaluate_by_ari_kw_str):
                    for ari in eval_ari_list:
                        FSS = self._calculate_fss_single_grid(qpf, qpe, fixed_radius, self.ari_grid_dict[ari])
                        fss_list.append(FSS)
                else:
                    for threshold in eval_threshold_list:
                        FSS = self._calculate_fss_single_grid(qpf, qpe, fixed_radius, threshold,
                                                              is_pctl_threshold = is_pctl_threshold,
                                                              pctl_over_eval_period = pctl_over_eval_period,
                                                              external_pctl_value_qpf = pctl_over_eval_period_dict[data_name].sel(pctl = threshold).item(),
                                                              external_pctl_value_qpe = pctl_over_eval_period_dict[self.truth_data_name].sel(pctl = threshold).item(),
                                                              include_zeros = include_zeros)
                        fss_list.append(FSS)
    
                fss_tmp_array = np.array(fss_list).reshape((1, fss_data_coords.shape[0]))
                if (v == 0):
                    fss_array = np.copy(fss_tmp_array) 
                else:
                    fss_array = np.concatenate((fss_array, fss_tmp_array), axis = 0)

                # Calculate frequency-related values based on the binary qpf and qpe fields
                # F_obs = fraction of observed points exceeding threshold over whole domain
                # F_model = fraction of forecast/model points exceeding threshold over whole domain
                # These values in turn are used to calculate AFSS, FSS_uniform, etc
                # Only do this for by_radius evaluations, since that's the context in which it  is most useful. 
                if (eval_type == evaluate_by_radius_kw_str):
                    if is_pctl_threshold:
                        if include_zeros:
                            threshold_for_F_model_calc = qpf.quantile(fixed_threshold/100.0)
                        else:
                            threshold_for_F_model_calc = qpf.where(qpf > 0.0).quantile(fixed_threshold/100.0)
                    else:
                        threshold_for_F_model_calc = self.fixed_fss_eval_threshold
                    binary_qpf = self._mask_data_array_based_on_threshold(qpf, threshold_for_F_model_calc)
                    F_model = np.where(binary_qpf)[0].shape[0]/binary_qpf.flatten().shape[0]
                    f_model_list.append(F_model)
                elif (eval_type == evaluate_by_radius_ari_threshold_kw_str):
                    binary_qpf = self._mask_data_array_based_on_threshold(qpf, self.fixed_ari_grid)
                    F_model = np.where(binary_qpf)[0].shape[0]/binary_qpf.flatten().shape[0]
                    f_model_list.append(F_model)

            # Convert numpy arrays to DataArrays
            if (data_name == self.truth_data_name):
                if (eval_type == evaluate_by_radius_kw_str) or (eval_type == evaluate_by_radius_ari_threshold_kw_str):
                    self.f_obs_da = xr.DataArray(f_obs_list, coords = [valid_dt_list], dims = [utils.period_end_time_dim_str])
                    self.f_obs_da.name = "observed_fractions"
                continue

            fss_da = xr.DataArray(fss_array, coords = [valid_dt_list, fss_data_coords], dims = [utils.period_end_time_dim_str, fss_data_dim_name])
            fss_da.name = "fss" 
            da_dict_fss[data_name] = fss_da
            
            if (eval_type == evaluate_by_radius_kw_str) or (eval_type == evaluate_by_radius_ari_threshold_kw_str):
                f_model_da = xr.DataArray(f_model_list, coords = [valid_dt_list], dims = [utils.period_end_time_dim_str])
                f_model_da.name = "forecast_fractions"
                da_dict_f_model[data_name] = f_model_da

            # Output to netCDF
            if write_to_nc:
                nc_out_fpath = self._configure_output_stats_nc_fpath(data_name, time_period_str, time_period_type = time_period_type, stat_type = stat_type)
                print(f"Writing {nc_out_fpath}")
                fss_da.to_netcdf(nc_out_fpath)

        # Here I decided to add the FSS dictionary as an attribute of the class
        # This will make it easier to access and manipulate the data in the dictionary in other methods.
        if (eval_type == evaluate_by_radius_kw_str) or (eval_type == evaluate_by_radius_ari_threshold_kw_str):
            self.fss_dict_by_radius = da_dict_fss
            self.f_model_dict = da_dict_f_model
        elif (eval_type == evaluate_by_ari_kw_str):
            self.fss_dict_by_ari = da_dict_fss
        else:
            self.fss_dict_by_threshold = da_dict_fss

        return da_dict_fss

    def how_to_calculate_aggregated_fss(self):
        print(f'calculate_aggregated_fss(external_fss_dict = None,\n'
              f'                         eval_type = [{evaluate_by_radius_kw_str}, {evaluate_by_threshold_kw_str},\n'
              f'                         {evaluate_by_radius_ari_threshold_kw_str}, {evaluate_by_ari_kw_str}],\n'
              f'                         time_period_type = "full_period")')

    def calculate_aggregated_fss(self, external_fss_dict = None, eval_type = evaluate_by_radius_kw_str, time_period_type = "full_period",
                                 is_pctl_threshold = False):
        if (external_fss_dict is not None):
            if (eval_type == evaluate_by_radius_kw_str) or (eval_type == evaluate_by_radius_ari_threshold_kw_str): # By radius
                self.fss_eval_radius_da = external_fss_dict[self.data_names[-1]].radius
                self.fss_dict_by_radius = external_fss_dict
                da_dict_fss = self.fss_dict_by_radius
            elif (eval_type == evaluate_by_ari_kw_str): # By ARI
                self.fss_eval_ari_da = external_fss_dict[self.data_names[-1]].ARI
                da_dict_fss = self.fss_dict_by_ari
            else: # By threshold
                if is_pctl_threshold:
                    self.fss_eval_threshold_da = external_fss_dict[self.data_names[-1]].pctl_threshold
                else:
                    self.fss_eval_threshold_da = external_fss_dict[self.data_names[-1]].threshold
                self.fss_dict_by_threshold = external_fss_dict
                da_dict_fss = self.fss_dict_by_threshold 
        else:
            if (eval_type == evaluate_by_radius_kw_str) or (eval_type == evaluate_by_radius_ari_threshold_kw_str):
                if not(hasattr(self, "fss_dict_by_radius")):
                    print("Error: You need to calculate FSS over varying evaluation radii first to perform aggregation")
                    return
                da_dict_fss = self.fss_dict_by_radius
            elif (eval_type == evaluate_by_ari_kw_str):
                if not(hasattr(self, "fss_dict_by_ari")):
                    print("Error: You need to calculate FSS over varying ARI grids first to perform aggregation")
                    return
                da_dict_fss = self.fss_dict_by_ari
            else:
                if not(hasattr(self, "fss_dict_by_threshold")):
                    print("Error: You need to calculate FSS over varying thresholds first to perform aggregation")
                    return
                da_dict_fss = self.fss_dict_by_threshold
        
        # Process time_period_type: list of date times, dimension name, etc.
        dtimes, dim_name, time_period_str = self._process_time_period_type_to_dtimes(time_period_type)

        # If aggregating seasonally, dtimes, which in this case will be a list of lists defining
        # seasonal datetime ranges, wasn't defined above.
        if (time_period_type == "seasonal"):
            dtimes = self._construct_season_dt_ranges(data_array)
        
        fss_agg_dict = {}
        afss_agg_dict = {}
        fss_uniform_agg_dict = {}
        for dtime in dtimes:
            da_dict_fss_each_dtime = {}
            da_dict_afss_each_dtime = {}
            da_dict_fss_uniform_each_dtime = {}
            for data_name, data_array in da_dict_fss.items():
                if (data_name == self.truth_data_name):
                    continue

                # Convert data coordinates to period beginning (much easier to aggregate over months that way)
                data_array = utils.convert_period_end_to_period_begin(data_array)
                data_to_aggregate = self._determine_agg_data_from_time_period_type(data_array, time_period_type, dtime)
                data = data_to_aggregate.mean(dim = utils.period_begin_time_dim_str)

                da_dict_fss_each_dtime[data_name] = data

                # Calculate AFSS and FSS_uniform (useful for plotting of by-radius aggregated FSS results)
                if (eval_type == evaluate_by_radius_kw_str) or (eval_type == evaluate_by_radius_ari_threshold_kw_str):
                    f_obs_da = utils.convert_period_end_to_period_begin(self.f_obs_da)
                    f_model_da = utils.convert_period_end_to_period_begin(self.f_model_dict[data_name])
                    
                    afss_da = (2 * f_obs_da * f_model_da)/(f_obs_da**2 + f_model_da**2)
                    fss_uniform_da = 0.5 * (1 + f_obs_da)

                    afss_data_to_aggregate = self._determine_agg_data_from_time_period_type(afss_da, time_period_type, dtime)
                    fss_uniform_data_to_aggregate = self._determine_agg_data_from_time_period_type(fss_uniform_da, time_period_type, dtime)

                    afss_data = afss_data_to_aggregate.mean().item()
                    fss_uniform_data = fss_uniform_data_to_aggregate.mean().item()

                    da_dict_afss_each_dtime[data_name] = afss_data
                    da_dict_fss_uniform_each_dtime[data_name] = fss_uniform_data

            fss_agg_dict[dtime] = da_dict_fss_each_dtime
            if  (eval_type == evaluate_by_radius_kw_str) or (eval_type == evaluate_by_radius_ari_threshold_kw_str):
                afss_agg_dict[dtime] = da_dict_afss_each_dtime
                fss_uniform_agg_dict[dtime] = da_dict_fss_uniform_each_dtime

        return fss_agg_dict, afss_agg_dict, fss_uniform_agg_dict 

    # Calculated occurence statistics aggregated over specified time periods (common monthly, common seasonal, etc.)
    def calculate_aggregated_occ_stats(self, which_stat = "all", time_period_type = "full_period", write_to_nc = False):
        # Process time_period_type: list of date times, dimension name, etc.
        dtimes, dim_name, time_period_str = self._process_time_period_type_to_dtimes(time_period_type)

        # If aggregating seasonally, dtimes, which in this case will be a list of lists defining
        # seasonal datetime ranges, wasn't defined above.
        if (time_period_type == "seasonal"):
            dtimes = self._construct_season_dt_ranges(data_array)
        
        occ_stats_agg_dict = {}
        for dtime in dtimes:
            da_dict_each_dtime = {}
            for data_name, data_array in self.da_dict.items(): 
                # Convert data coordinates to period beginning (much easier to aggregate over months that way)
                data_array = utils.convert_period_end_to_period_begin(data_array)
                da_dict_each_dtime[data_name] = self._determine_agg_data_from_time_period_type(data_array, time_period_type, dtime)
                
            all_occ_stats_dict = self.calculate_occ_stats(da_dict_each_dtime)
            if (which_stat == "all"):
                occ_stats_agg_dict[dtime] = all_occ_stats_dict 
            else:
                stat_dict = self.extract_occ_stat_dict(all_occ_stats_dict, which_stat)
                occ_stats_agg_dict[dtime] = stat_dict 

        # Output to netCDF
        if write_to_nc:
            for data_name in self.data_names:
                if (data_name == self.truth_data_name):
                    continue

                if (which_stat == "all"):
                    out_da_dict = {"hits": [], "misses": [], "false_alarms": [], "correct_negatives": [],
                                   "total_events": [], "CSI": [], "ETS": [], "frequency_bias": []}
                    stat_type = "occ_stats"
                else:
                    out_da_dict = {which_stat: []}
                    stat_type = which_stat 

                for dtime in dtimes:
                    for stat_name, da_list in out_da_dict.items():
                        if (which_stat == "all"):
                            out_da_dict[stat_name].append(self.extract_occ_stat_data_array(occ_stats_agg_dict[dtime], stat_name, data_name))
                        else:
                            out_da_dict[stat_name].append(occ_stats_agg_dict[dtime][data_name])

                for stat_name, da_list in out_da_dict.items():
                    out_da = xr.concat(da_list, dim = dim_name)
                    if (time_period_type == "seasonal"):
                        out_da.coords[dim_name] = self._construct_seasonal_dt_str_list(dtimes) 
                    else:
                        out_da.coords[dim_name] = dtimes 
                    out_da_dict[stat_name] = out_da 

                out_ds = xr.Dataset(out_da_dict)
                nc_out_fpath = self._configure_output_stats_nc_fpath(data_name,
                                                                     time_period_str,
                                                                     time_period_type = time_period_type,
                                                                     stat_type = stat_type)
                print(f"Writing {nc_out_fpath}")
                out_ds.to_netcdf(nc_out_fpath)

        return occ_stats_agg_dict
    
    def sum_data_over_full_time_period(self):
        print("Calculating data summed over full time period")
        self.da_dict_summed_time = {} 
        for data_name, da in self.da_dict.items():
            # Keep time dimension in summmed data array so we are able to operate on this
            # dimension in various plotting method and related helper methods.
            # Also set min_count to the shape of the DataArray's time dimension, which requires
            # that ALL data along this dimension be non-NaN to take the sum; otherwise, NaN is returned
            da_summed_time = da.sum(dim = utils.period_end_time_dim_str,
                                    keepdims = True,
                                    skipna = True,
                                    min_count = da.coords[utils.period_end_time_dim_str].shape[0])
            end_dt = pd.Timestamp(da[utils.period_end_time_dim_str].values[-1])
            da_summed_time.coords[utils.period_end_time_dim_str] = [end_dt]
 
            # Set attributes properly
            num_time_intervals = da[utils.period_end_time_dim_str].shape[0]
            total_time_period_hours = num_time_intervals * self.temporal_res
            pdp.add_attributes_to_data_array(da_summed_time,
                                               short_name = f"{total_time_period_hours}-hour precipitation", 
                                               long_name = f"Precipitation accumulated over the prior {total_time_period_hours} hour(s)",
                                               units = da.units,
                                               interval_hours = total_time_period_hours) 
            self.da_dict_summed_time[data_name] = da_summed_time 
    ##### END Public methods stats calculations #####

    ##### Private methods to support stats calculations #####
    # Based on time_period_type ('monthly', etc.), determine list of dtimes, dimension names, and time period string
    def _process_time_period_type_to_dtimes(self, time_period_type):
        match time_period_type:
            case None: # Use the time coordinates as they exist in the data arrays without any further temporal aggregation
                dtimes = self.valid_dt_list
                dim_name = utils.period_end_time_dim_str
                time_period_str = self.daily_time_period_str
            case "daily":
                dtimes = self.valid_daily_dt_list_period_begin # Need period beginning times for daily data
                dim_name = utils.days_dim_str 
                time_period_str = self.daily_time_period_str_period_begin
            case "monthly":
                dtimes = self.valid_monthly_dt_list
                dim_name = utils.months_dim_str 
                time_period_str = self.monthly_time_period_str
            case "seasonal":
                dtimes = []  # dtimes list defined in methods that use this private method 
                dim_name = utils.seasons_dim_str
                time_period_str = self.monthly_time_period_str 
            case "annual":
                dtimes = self.valid_annual_dt_list
                dim_name = utils.annual_dim_str 
                time_period_str = self.annual_time_period_str
            case "full_period":
                dtimes = [ self.monthly_time_period_str ] 
                dim_name = utils.full_period_dim_str
                time_period_str = self.monthly_time_period_str
            case "common_monthly":
                dtimes = ppu.construct_monthly_string_list()
                dim_name = utils.common_month_dim_str 
                time_period_str = self.monthly_time_period_str 
            case "common_seasonal":
                dtimes = ppu.construct_seasonal_string_list() 
                dim_name = utils.common_season_dim_str
                time_period_str = self.monthly_time_period_str 
            case _:
                print(f"Error: Unrecognized time period type {time_period_type}")
                return

        return dtimes, dim_name, time_period_str

    def _convert_occ_stats_np_array_to_data_array(self, np_array, data_coords, stat_name):
        # Convert numpy array to DataArray
        da = xr.DataArray(np_array, coords = [data_coords], dims = ["threshold"])
        da.name = stat_name 

        return da

    # Returns a list of date-time ranges corresponding to each season valid within
    # the time dimension of the input data array.
    def _construct_season_dt_ranges(self, da):
        _, time_dim, _ = self._create_datetime_list_from_da_time_dim(da)

        all_dtimes = [pd.Timestamp(i) for i in da[time_dim].values]
        current_dt =  all_dtimes[0]
        
        season_dt_ranges = []
        while (utils.datetime2unix(current_dt) <= utils.datetime2unix(all_dtimes[-1])):
            if (current_dt.month < 3):
                start_dt_next_season = pd.Timestamp(current_dt.year, 3, 1)
            elif (current_dt.month < 6):
                start_dt_next_season = pd.Timestamp(current_dt.year, 6, 1)
            elif (current_dt.month < 9):
                start_dt_next_season = pd.Timestamp(current_dt.year, 9, 1)
            elif (current_dt.month < 12):
                start_dt_next_season = pd.Timestamp(current_dt.year, 12, 1)
            else: # December case: next season starts in March of NEXT year
                start_dt_next_season = pd.Timestamp(current_dt.year + 1, 3, 1)

            # Since slicing will be inclusive, the dt range should end on the final day of the current season
            season_dt_range = [current_dt, start_dt_next_season - pd.Timedelta(days = 1)]
            season_dt_ranges.append(season_dt_range)
            current_dt = start_dt_next_season

        return season_dt_ranges

    # Based on the time period type (e.g., 'monthly', 'common seasonal', etc.), and a given dtime,
    # filter a DataArray to only data within the current dtime, to be subsequently aggregated.
    def _determine_agg_data_from_time_period_type(self, data_array, time_period_type, dtime):
        match time_period_type:
            case None:
                dtime_str = f"{dtime:%Y-%m-%d %H:%M:%S}"
                data_to_aggregate = data_array.sel(period_end_time = dtime_str)
                # If working with daily (as opposed to sub-daily) data, the .sel call above will remove the time dimension. Use expand_dims to add it back.
                if (len(data_to_aggregate.shape) == 2):
                    data_to_aggregate = data_to_aggregate.expand_dims(dim = {utils.period_begin_time_dim_str: [dtime]})
            case "daily":
                dtime_str = f"{dtime:%Y-%m-%d}"
                data_to_aggregate = data_array.sel(period_begin_time = dtime_str)
                if (len(data_to_aggregate.shape) == 2):
                    data_to_aggregate = data_to_aggregate.expand_dims(dim = {utils.period_begin_time_dim_str: [dtime]})
            case "monthly":
                dtime_str = f"{dtime:%Y-%m}"
                data_to_aggregate = data_array.sel(period_begin_time = dtime_str)
            case "seasonal":
                data_to_aggregate = data_array.sel(period_begin_time = slice(dtime[0], dtime[1]))
            case "annual":
                dtime_str = f"{dtime:%Y}"
                data_to_aggregate = data_array.sel(period_begin_time = dtime_str)
            case "full_period":
                data_to_aggregate = data_array
            case "common_monthly":
                data_to_aggregate = self._select_data_by_common_time_period(data_array, dtime)
            case "common_seasonal":
                data_to_aggregate = self._select_data_by_common_time_period(data_array, dtime)
            case _:
                raise NotImplementedError

        return data_to_aggregate

    # Select data from a data array for the same month or same season (e.g., all January's) 
    def _select_data_by_common_time_period(self, data_array, time_period):
        if (type(time_period) is str):
            match time_period.lower()[:3]:
                case "jan":
                    month_list = [1]
                case "feb":
                    month_list = [2]
                case "mar":
                    month_list = [3]
                case "apr":
                    month_list = [4]
                case "may":
                    month_list = [5]
                case "jun":
                    month_list = [6]
                case "jul":
                    month_list = [7]
                case "aug":
                    month_list = [8]
                case "sep":
                    month_list = [9]
                case "oct":
                    month_list = [10]
                case "nov":
                    month_list = [11]
                case "dec":
                    month_list = [12]
                case "djf":
                    month_list = [12,1,2]
                case "mam":
                    month_list = [3,4,5]
                case "jja":
                    month_list = [6,7,8]
                case "son":
                    month_list = [9,10,11]
                case _:
                    print(f"Time period string is {time_period}, not a valid month or season; not selecting data from data array")
                    return data_array
        else:
            print(f"Invalid time period string {time_period}; not selecting data from data array;\n"
                   "must be a string representing the name of a month, the first three letters of a month\n"
                   "(e.g., Aug for August; case insensitive), or three letters representing a season\n"
                   "(options are DJF, MAM, JJA, SON); case insensitive")
            return data_array

        # Convert time coordinates to period beginning so that accumulated precip ending at, for example, 00z Feb 1
        # is interpreted as valid *during* January.
        if (utils.period_end_time_dim_str in data_array.dims): 
            data_array = utils.convert_period_end_to_period_begin(data_array)

        da_sel_time_period = data_array.sel(period_begin_time = data_array.period_begin_time.dt.month.isin(month_list))
        return da_sel_time_period

    # Construct strings of the current season and the year to which it
    # corresponds (e.g., Decemember corresponds to DJF of the following year)
    def _construct_seasonal_dt_str_list(self, dtime_ranges):
        seasonal_dt_str_list = []
        for dtime_range in dtime_ranges:
            if (dtime_range[0].month < 3): # Winter
                season_str = "DJF"
                year_str = f"{dtime_range[0]:%y}"
            elif (dtime_range[0].month < 6): # Spring 
                season_str = "MAM"
                year_str = f"{dtime_range[0]:%y}"
            elif (dtime_range[0].month < 9): # Summer
                season_str = "JJA"
                year_str = f"{dtime_range[0]:%y}"
            elif (dtime_range[0].month < 12): # Fall
                season_str = "SON"
                year_str = f"{dtime_range[0]:%y}"
            else: # December, so Winter of the NEXT year
                season_str = "DJF"
                year_str = f"{dtime_range[-1]:%y}"

            seasonal_dt_str_list.append(f"{season_str}{year_str}")
        return seasonal_dt_str_list

    # Convert DataArrays to binary (1/0) values, based on threshold; used in FSS calculation.
    # Set data array to 1 at or above threshold, zero below it
    # NOTE: this function takes an xarray DataArray as input and returns a numpy array.
    def _mask_data_array_based_on_threshold(self, da, threshold):
        # Take the arrays down to 2-D, removing the time dimension
        if (len(da.shape) == 3):
            da = da[0,:,:]

        return np.where(da >= threshold, 1, 0) 

    # Create circular footprint for FSS calculation
    # Potential FIXME: For the expected behavior to be obtained, <radius> must be
    # evenly divisible by <grid_cell_size>, e.g., radius = 0.5 degrees; grid_cell_size = 0.25 degrees.
    # Specifically, if this is not the case, the function will return a square of all 1s in some cases,
    # rather than a square of 0s circumscribing a circle of 1s. This may be OK (won't fix) because it
    # doesn't make sense to use a radius that's equivalent to a non-integer number of grid cells.
    def _get_footprint_for_fss(self, radius):
        radius_number_grid_cells = int(radius/self.grid_cell_size)

        # In this step, we obtain a square of zeros (with side length, in number of grid cells, of radius_number_grid_cells * 2 + 1)
        # circumscribing a circle of ones (with radius, in number of grid cells, of radius_number_grid_cells):

        # 1) Create a footprint: just an array of 1s
        footprint = (np.ones((radius_number_grid_cells * 2 + 1, radius_number_grid_cells * 2 + 1))).astype(int)

        # 2) Set the centerpoint of the array to zero (needed for the subsequent distance calculation)
        footprint[math.ceil(radius_number_grid_cells), math.ceil(radius_number_grid_cells)] = 0

        # 3) Within the footprint, calculate each point's distance from the center point
        dist = scipy.ndimage.distance_transform_edt(footprint, sampling = [self.grid_cell_size, self.grid_cell_size])

        # 4) Set the footprint to zeros where distance calculated in step 3) is greater than radius; keep it
        # set to one where distance is less than radius, obtaining the square of zeros circumscribing the circle of ones 
        return np.where(np.greater(dist, radius), 0, 1)

    # Calculate FSS for a single spatial grid (i.e., at a single valid time).
    # Code from Craig Schwartz via Trevor Alcott. See 20250130 email from Trevor
    # which is part of thread entitled "Experience with fractions skill score?"
    # FIXME (potentially): how are NaNs being handled? I think they are being converted to zeros by _mask_data_array_based_on_threshold
    # which may not be desirable. We should keep them as NaNs, but then how will that affect the FSS calculation here?
    def _calculate_fss_single_grid(self, qpf, qpe, radius, threshold,
                                   is_pctl_threshold = False,
                                   include_zeros = False,
                                   pctl_over_eval_period = False,
                                   external_pctl_value_qpf = None,
                                   external_pctl_value_qpe = None):
        # Calculate footprint, i.e., evaluation area
        footprint = self._get_footprint_for_fss(radius)
       
        # Convert qpf and qpe arrays to numpy arrays containing 1s and 0s based on
        # whether precipitation amount is at or above (set to 1) or below (set to 0) <threshold>.
        if is_pctl_threshold:
            if pctl_over_eval_period and \
               (external_pctl_value_qpf is not None) and \
               (external_pctl_value_qpe is not None):
                threshold_amount_qpf = external_pctl_value_qpf
                threshold_amount_qpe = external_pctl_value_qpe
            else:
                if not(include_zeros):
                    threshold_amount_qpf = qpf.where(qpf > 0.0).quantile(threshold/100.0)
                    threshold_amount_qpe = qpe.where(qpe > 0.0).quantile(threshold/100.0)
                else:
                    threshold_amount_qpf = qpf.quantile(threshold/100.0) 
                    threshold_amount_qpe = qpe.quantile(threshold/100.0)

            binary_qpf = self._mask_data_array_based_on_threshold(qpf, threshold_amount_qpf)
            binary_qpe = self._mask_data_array_based_on_threshold(qpe, threshold_amount_qpe)
        else: 
            binary_qpf = self._mask_data_array_based_on_threshold(qpf, threshold)
            binary_qpe = self._mask_data_array_based_on_threshold(qpe, threshold)

        # Calculate forecast_fractions and observed fractions terms in the FSS formula.
        # These are the M (model) and O (observed) terms calculated in Roberts and Lean (2008)
        # equations 2 and 3, and what Trevor's code refers to as pf and po, respectively.
        # CONCEPTUAL PROCEDURE:
            # For every grid point, calculate the number of points within the footprint centered on the grid point
            # for which the binary_qpf and binary_qpe arrays equal 1 (i.e., <qpf> and <qpe> are at or above <threshold>).
            # Divide by the size of the footprint [np.sum(footprint)] to convert this number of points to a spatial
            # fraction of the footprint.
        # IMPLEMENTATION using fftconvolve:
            # I don't yet understand how fftconvolve calculates the number of points equal to one in each grid
            # point's neighborhood other than to state that it uses a Fast Fourier Transform (FFT) technique. 
        forecast_fractions = np.around(scipy.signal.fftconvolve(binary_qpf, footprint, mode = "same"))/np.sum(footprint)
        observed_fractions = np.around(scipy.signal.fftconvolve(binary_qpe, footprint, mode = "same"))/np.sum(footprint)

        # Calculate gridsize (Nx * Ny)
        gridsize = np.shape(binary_qpe)[0] * np.shape(binary_qpe)[1]

        # Calculate numerator [Equation 5 in Roberts and Lean (2008)]
        # which is the mean squared error (MSE) of the forecast fractions (forecast_fractions)
        # compared to the observed fractions (observed_fractions)
        mse = 1/gridsize * np.sum((forecast_fractions - observed_fractions)**2)

        # Calculate denominator [Equation 7 in Roberts and Lean (2008)]
        # which is the mean squared error (MSE) of a low-skill reference forecast
        mse_reference = 1/gridsize * (np.sum(forecast_fractions**2) + np.sum(observed_fractions**2))

        if (mse_reference > 0):
            return 1.0 - float(mse)/float(mse_reference)
        else:
            return np.nan 
    
    def _open_ari_threshold_grid(self, ari, duration):
        ari_nc_dir = os.path.join(self.input_dir, f"ARIs.{self.data_grid_name}")
        data_name = f"ARI.{self.data_grid_name}.{ari:04d}_year.{duration:03d}_hour_precipitation"
        ari_fname = f"{data_name}.nc"
        ari_fpath = os.path.join(ari_nc_dir, ari_fname)
        if not(os.path.exists(ari_fpath)):
            print(f"Error: ARI grid file {ari_fpath} does not exist")
            sys.exit(1)
       
        print(f"Reading ARI grid file {ari_fpath}") 
        ari_grid = xr.open_dataset(ari_fpath).precip  
        ari_grid = self._subset_data_to_region(ari_grid, data_name = data_name)

        return ari_grid

    def _set_output_var_name(self, data_array):
        if (self.temporal_res == "native"):
            output_var_name = "precipitation"
        else:
            output_var_name = f"precipitation_{self.temporal_res}_hour"
        data_array.name = output_var_name

    def _configure_output_stats_nc_fpath(self, data_name, time_period_str, time_period_type = None, stat_type = None, agg_type = None):
        if (data_name == self.data_grid):
            main_prefix = f"{data_name}.{self.native_grid_name}.{self.temporal_res:02d}_hour_precipitation"
            dir_name = f"{main_prefix}.stats"
        else:
            main_prefix = f"{data_name}.{self.data_grid_name}.{self.temporal_res:02d}_hour_precipitation"
            dir_name = f"{main_prefix}.stats"

        nc_dir = os.path.join(self.input_dir, dir_name)
        if (not os.path.exists(nc_dir)):
            os.mkdir(nc_dir)

        if (time_period_type is not None):
            main_prefix += f".{time_period_type}"

        if (stat_type is not None):
            main_prefix += f".{stat_type}"

        if (agg_type is not None):
            main_prefix += f".{agg_type}"
        
        fname = f"{main_prefix}.{time_period_str}.{self.region}.nc"
        fpath = os.path.join(nc_dir, fname)
        return fpath
    ##### END Private methods to support stats calculations #####

    ##### Public methods plotting #####
    def create_da_dict_single_time_for_case_study(self, dt_str, time_dim_name = utils.period_end_time_dim_str):
        case_da_dict = {}
        for data_name, da in self.da_dict.items():
          case_da_dict[data_name] = da.sel(period_end_time = dt_str).expand_dims(dim = {time_dim_name: [dt_str]})

        return case_da_dict

    def how_to_plot_aggregated_occ_stats_by_threshold(self):
        print('plot_aggregated_occ_stats_by_threshold(occ_stats_dict, which_stat = "CSI", time_period_type = "full_period",\n'
               '                                      xaxis_explicit_values = False, plot_levels = None)')

    def plot_aggregated_occ_stats_by_threshold(self, occ_stats_dict, which_stat = "CSI", time_period_type = "full_period",
                                               xaxis_explicit_values = False, plot_levels = None):
        # Based on this particular dataset, get a list of all the valid datetimes we're going to plot
        dtimes = sorted(list(occ_stats_dict.keys()))

        # Loop through dtimes, creating a plot for each one 
        for dtime in dtimes: 
            if (type(dtime) is pd.Timestamp) or (type(dtime) is dt.datetime):
                dt_str = dtime.strftime("%Y%m") 
            elif (type(dtime) is str):
                dt_str = dtime
            else:
                dt_str = dtime.strftime("%Y%m")

            if (dt_str in ppu.construct_monthly_string_list()):
                time_period_number = ppu.month_string_to_month_number(dt_str)
                dt_str_ext = f"{time_period_number:02d}{dt_str}.{self.monthly_time_period_str}"
            elif (dt_str in ppu.construct_seasonal_string_list()): 
                time_period_number = ppu.season_string_to_season_number(dt_str)
                dt_str_ext = f"{time_period_number:02d}{dt_str}.{self.monthly_time_period_str}"
            else:
                dt_str_ext = dt_str

            # Create figure
            fig = plt.figure(figsize = (13, 10))
            axis = plt.gca() 
            
            # Set axes limits and ticks
            if xaxis_explicit_values: # Plot explicitly against the selected eval radii or thresholds, with each value evenly spaced on x-axis
                xaxis_var = np.arange(self.threshold_da_for_occ_stats.shape[0])
                xticks = self.threshold_da_for_occ_stats
                axis.set_xticks(xaxis_var, xticks)
                axis.set_xlim(xaxis_var[0], xaxis_var[-1])
            else:
                xaxis_var = self.threshold_da_for_occ_stats 
                xticks = np.arange(0, xaxis_var[-1] + 10, 10)
                axis.set_xticks(xticks)
                axis.set_xlim(0, xaxis_var[-1])
            if (plot_levels is not None):
                axis.set_ylim(plot_levels[0], plot_levels[-1])
                axis.set_yticks(plot_levels)
            else:
                if (which_stat != "frequency_bias"):
                    axis.set_ylim(0, 1.0)
                    axis.set_yticks(np.arange(0, 1.1, 0.1)) 
                else:
                    axis.set_ylim(0, 2.0)
                    axis.set_yticks(np.arange(0, 2.2, 0.2)) 
            axis.tick_params(axis = "both", labelsize = 15)
            axis.grid(True, linewidth = 1.5)

            # Add title and axes labels 
            short_name = pdp.format_short_name(self.da_dict[self.truth_data_name])
            plt.title(f"{which_stat} vs. threshold, {self.region} {short_name}: {dt_str}", size = 15)
            plt.xlabel("Threshold (mm)", size = 15)
            if (which_stat == "frequency_bias"):
                plt.ylabel("Frequency Bias", size = 15) 
            else:
                plt.ylabel(which_stat, size = 15) 

            # Plot data
            if (which_stat == "frequency_bias"): # For frequency bias, add a line at bias = 1 (unbiased forecast)
                axis.plot([0, xaxis_var[-1]], [1, 1], linewidth = 3, color = "black") 
            single_occ_stat_dict = self.extract_occ_stat_dict(occ_stats_dict[dtime], which_stat)
            for data_name, da in single_occ_stat_dict.items():
                if (data_name == self.truth_data_name):
                    continue
                axis.plot(xaxis_var, da, linewidth = 2.5, label = data_name,
                          color = ppu.datasets_colors_dict[data_name])
            axis.legend(loc = "best", prop = {"size": 15})

            # Save figure 
            fig.tight_layout()
            fig_name = f"{which_stat}_threshold_mm.{self.data_names_str}{time_period_type}.{short_name}.{dt_str_ext}.{self.region}.png"
            fig_path = os.path.join(self.plot_output_dir, fig_name)
            print(f"Saving {fig_path}")
            plt.savefig(fig_path)

    def how_to_plot_aggregated_fss(self):
        print(f'plot_aggregated_fss(da_dict = None,\n'
              f'                    eval_type = [{evaluate_by_radius_kw_str}, {evaluate_by_threshold_kw_str}, {evaluate_by_radius_ari_threshold_kw_str}, {evaluate_by_ari_kw_str}],\n'
              f'                    xaxis_explicit_values = False,\n'
              f'                    xaxis_var_ticks = None\n'
              f'                    time_period_type = "full_period",\n'
              f'                    plot_levels_fss = None,\n'
              f'                    plot_levels_frequency_bias = None,\n'
              f'                    is_pctl_threshold = False,\n'
              f'                    include_frequency_bias = False,\n'
              f'                    include_fss_uniform = False,\n'
              f'                    fontsize = 15,\n'
              f'                    linewidth = 2.5)')

    def plot_aggregated_fss(self,
                            da_dict = None,
                            eval_type = evaluate_by_radius_kw_str,
                            xaxis_explicit_values = False,
                            xaxis_var_ticks = None,
                            time_period_type = "full_period",
                            plot_levels_fss = None,
                            plot_levels_frequency_bias = None,
                            is_pctl_threshold = False,
                            include_frequency_bias = False,
                            include_fss_uniform = False,
                            fontsize = 15,
                            linewidth = 2.5):
        if (da_dict is None):
            da_dict = self.da_dict

        # Only plot frequency bias on second axis if the first axis is plotted against amount thresholds.
        if is_pctl_threshold or (eval_type != evaluate_by_threshold_kw_str):
            include_frequency_bias = False

        # Aggregated FSS data to plot
        fss_agg_dict, afss_agg_dict, fss_uniform_agg_dict = self.calculate_aggregated_fss(eval_type = eval_type, time_period_type = time_period_type)

        # Frequency bias data to plot (done for plotting against thresholds, only)
        if include_frequency_bias:
            frequency_bias_dict = self.calculate_aggregated_occ_stats(which_stat = "frequency_bias", time_period_type = time_period_type)
        
        # Based on this particular dataset, get a list of all the valid datetimes we're going to plot
        dtimes = sorted(list(fss_agg_dict.keys()))
        
        # Loop through dtimes, creating a plot for each one 
        for dtime in dtimes: 
            if (type(dtime) is pd.Timestamp) or (type(dtime) is dt.datetime):
                dt_str = dtime.strftime("%Y%m") 
            elif (type(dtime) is str):
                dt_str = dtime
            else:
                dt_str = dtime.strftime("%Y%m")

            if (dt_str in ppu.construct_monthly_string_list()):
                time_period_number = ppu.month_string_to_month_number(dt_str)
                dt_str_ext = f"{time_period_number:02d}{dt_str}.{self.monthly_time_period_str}"
            elif (dt_str in ppu.construct_seasonal_string_list()): 
                time_period_number = ppu.season_string_to_season_number(dt_str)
                dt_str_ext = f"{time_period_number:02d}{dt_str}.{self.monthly_time_period_str}"
            else:
                dt_str_ext = dt_str

            # Determine x-axis ticks
            if (eval_type == evaluate_by_radius_kw_str):
                if xaxis_explicit_values: # Plot explicitly against the selected eval radii or thresholds, with each value evenly spaced on x-axis
                    xaxis_var = np.arange(self.fss_eval_radius_da.shape[0])
                    xticks = self.fss_eval_radius_da.values
                else:
                    xaxis_var = self.fss_eval_radius_da 
                    xticks = np.arange(0, xaxis_var[-1] + 1, 1)
            elif (eval_type == evaluate_by_radius_ari_threshold_kw_str): 
                if xaxis_explicit_values:
                    xaxis_var = np.arange(self.fss_eval_radius_da.shape[0])
                    xticks = self.fss_eval_radius_da.values
                else:
                    xaxis_var = self.fss_eval_radius_da 
                    xticks = np.arange(0, xaxis_var[-1] + 1, 1)
            elif (eval_type == evaluate_by_ari_kw_str):
                if xaxis_explicit_values: # Plot explicitly against ARI thresholds, with each value evenly spaced on x-axis
                    xaxis_var = np.arange(self.fss_eval_ari_da.shape[0])
                    xticks = self.fss_eval_ari_da.values
                else:
                    xaxis_var = self.fss_eval_ari_da 
                    xticks = np.arange(0, xaxis_var[-1] + 1, 1)
            else: # Evaluate by threshold 
                if xaxis_explicit_values:
                    xaxis_var = np.arange(self.fss_eval_threshold_da.shape[0])
                    xticks = self.fss_eval_threshold_da.values
                else:
                    xaxis_var = self.fss_eval_threshold_da
                    xticks = np.arange(0, xaxis_var[-1] + 10, 10)
            xlims = (xaxis_var[0], xaxis_var[-1])
           
            # If xaxis_var_ticks is not None, override previous settings for xaxis_explicit values, xticks, and xlims.
            # Do this as a separate if-block to ensure that xaxis_var is still set above (xaxis_var is the
            # array that is plotted against below).
            if (xaxis_var_ticks is not None):
                xaxis_explicit_values = False
                xticks = xaxis_var_ticks
                xlims = (xaxis_var_ticks[0], xaxis_var_ticks[-1])

            # Determine y-axis ticks
            if (plot_levels_fss is not None):
                ylims = (plot_levels_fss[0], plot_levels_fss[-1])
                yticks = plot_levels_fss
            else:
                ylims = (0, 1.0)
                yticks = np.arange(0, 1.1, 0.1)

            # Create figure; set figure title; axis labels and figure name
            short_name = pdp.format_short_name(da_dict[self.truth_data_name])
            if (eval_type == evaluate_by_radius_kw_str):
                if is_pctl_threshold:
                    fixed_threshold_units = "th pctl"
                    fixed_threshold_units_no_space = fixed_threshold_units.replace(" ", "_")
                    pctl_over_eval_period_str = self.pctl_over_eval_period_str 
                else:
                    fixed_threshold_units = " mm"
                    fixed_threshold_units_no_space = fixed_threshold_units.replace(" ", "")
                    pctl_over_eval_period_str = "" 
                title = f"FSS vs. radius, {self.region} {short_name} (t = {self.fixed_fss_eval_threshold:0.1f}{fixed_threshold_units}): {dt_str}"
                xlabel = f"Evaluation radius ({self.fss_eval_radius_units})" 
                fig_name = f"FSSradius.{self.data_names_str}thresh{self.fixed_fss_eval_threshold:0.1f}{fixed_threshold_units_no_space}{pctl_over_eval_period_str}.{time_period_type}.{short_name}.{dt_str_ext}.{self.region}.png"
            elif (eval_type == evaluate_by_radius_ari_threshold_kw_str): 
                title = f"FSS vs. radius, {self.region} {short_name} (t = {self.fixed_fss_eval_threshold:04d} year ARI): {dt_str}"
                xlabel = f"Evaluation radius ({self.fss_eval_radius_units})" 
                fig_name = f"FSSradius.{self.data_names_str}thresh{self.fixed_fss_eval_threshold:04d}year_ari.{time_period_type}.{short_name}.{dt_str_ext}.{self.region}.png"
            elif (eval_type == evaluate_by_ari_kw_str):
                title = f"FSS of ARI exceedances, {self.region} {short_name} (r = {self.fixed_fss_eval_radius:0.2f} {self.fss_eval_radius_units}): {dt_str}"
                xlabel = f"ARI (years)" 
                fig_name = f"FSSari.{self.data_names_str}radius{self.fixed_fss_eval_radius:0.2f}{self.fss_eval_radius_units}.{time_period_type}.{short_name}.{dt_str_ext}.{self.region}.png"
            else: # Evaluate by threshold
                title_prefix = "FSS"
                figname_prefix = "FSS"
                if include_frequency_bias:
                    title_prefix += ", frequency bias"
                    figname_prefix += ".freqBias."
                if is_pctl_threshold:
                    threshold_units = "pctl"
                    pctl_over_eval_period_str = self.pctl_over_eval_period_str 
                else:
                    threshold_units = "mm"
                    pctl_over_eval_period_str = "" 
                title = f"{title_prefix} vs. threshold, {self.region} {short_name} (r = {self.fixed_fss_eval_radius:0.2f} {self.fss_eval_radius_units}): {dt_str}"
                xlabel = f"Threshold ({threshold_units})" 
                fig_name = f"{figname_prefix}threshold_{threshold_units}{pctl_over_eval_period_str}.{self.data_names_str}radius{self.fixed_fss_eval_radius:0.2f}{self.fss_eval_radius_units}.{time_period_type}.{short_name}.{dt_str_ext}.{self.region}.png"
            fig = plt.figure(figsize = (13, 10))

            plot_dicts_list = [ fss_agg_dict[dtime] ]
            ylabels_list = ["Fractions Skill Score (FSS)"]
            ylims_list = [ ylims ]
            yticks_list = [ yticks ]
            subplot_titles_list = ["FSS"]
            if include_frequency_bias:
                axes_list = [
                            plt.subplot2grid((1, 2), (0, 0), colspan = 1, rowspan = 1),
                            plt.subplot2grid((1, 2), (0, 1), colspan = 1, rowspan = 1),
                            ]
                plot_dicts_list.append(frequency_bias_dict[dtime])
                ylabels_list.append("Frequency Bias")
                if (plot_levels_frequency_bias is not None):
                    ylims_list.append( (plot_levels_frequency_bias[0], plot_levels_frequency_bias[-1]) )
                    yticks_list.append(plot_levels_frequency_bias)
                else:
                    ylims_list.append( (0, 2.0) )
                    yticks_list.append( np.arange(0, 2.2, 0.2) )
                subplot_titles_list.append("Frequency Bias")
            else:
                axes_list = [
                            plt.subplot2grid((1, 1), (0, 0)),
                            ]

            # Plot data
            for axis, plot_dict, ylabel, ylims, yticks, subplot_title in zip(axes_list, plot_dicts_list, ylabels_list, ylims_list, yticks_list, subplot_titles_list):
                axis.set_xlabel(xlabel, size = fontsize)
                axis.set_ylabel(ylabel, size = fontsize)
                axis.set_xlim(xlims)
                axis.set_ylim(ylims)
                if xaxis_explicit_values:
                    axis.set_xticks(xaxis_var, xticks)
                else:
                    axis.set_xticks(xticks)
                axis.set_yticks(yticks) 
                axis.tick_params(axis = "both", labelsize = fontsize)
                axis.grid(True, linewidth = 1.5)

                for data_name, da in plot_dict.items():
                    if (data_name == self.truth_data_name):
                        continue
                    axis.plot(xaxis_var, da, linewidth = linewidth, label = data_name,
                              color = ppu.datasets_colors_dict[data_name])
                if (include_fss_uniform) and \
                (subplot_title == "FSS") and \
                ((eval_type == evaluate_by_radius_kw_str) or (eval_type == evaluate_by_radius_ari_threshold_kw_str)):
                    fss_uniform = fss_uniform_agg_dict[dtime][self.data_names[-1]]
                    axis.plot([0, xticks[-1]], [fss_uniform, fss_uniform], linewidth = 3, color = "black", linestyle = "dashed") 
                if include_frequency_bias:
                    axis.set_title(subplot_title, fontsize = fontsize)
                    if (subplot_title == "Frequency Bias"): # If we're working on the frequency bias axis, add a line at bias = 1 (unbiased forecast)
                        axis.plot([0, xaxis_var[-1]], [1, 1], linewidth = 3, color = "black") 
                axis.legend(loc = "best", prop = {"size": fontsize})

            # Save figure 
            fig.suptitle(title, size = fontsize)
            fig.tight_layout()
            fig_path = os.path.join(self.plot_output_dir, fig_name)
            print(f"Saving {fig_path}")
            plt.savefig(fig_path)

    def how_to_plot_fss_timeseries(self):
        print("plot_fss_timeseries(eval_radius, eval_threshold, plot_levels = None)")

    # Plot time series of FSS for each valid time (i.e., each individual grid evaluation)
    def plot_fss_timeseries(self, eval_radius, eval_threshold, plot_levels = None):
        if not(hasattr(self, "fss_dict_by_radius")) or not(hasattr(self, "fss_dict_by_threshold")):
            print("Error: You need to calculate FSS over varying evaluation radii and thresholds")
            return
        
        if (eval_threshold not in self.fss_eval_threshold_da):
            print(f"Error: FSS not calculated for threshold {eval_threshold:0.1f}")
            return

        if (eval_radius != self.fixed_fss_eval_radius) and (eval_threshold != self.fixed_fss_eval_threshold):
            print(f"Error: FSS not calculated for radius {eval_radius:0.1f} (at threshold {eval_threshold})")
            return

        # Create figure 
        plt.figure(figsize = (15, 10))
        short_name = pdp.format_short_name(self.da_dict[self.truth_data_name])
        plt.title(f"FSS time series, {self.region} {short_name} (r = {eval_radius:0.2f} {self.fss_eval_radius_units}, t = {eval_threshold:0.1f} {self.fss_eval_threshold_units}): {self.daily_time_period_str}", size = 15)
        plt.xlabel("Period end time", size = 15)
        plt.ylabel("Fractions Skill Score (FSS)", size = 15)
        plt.xlim(self.valid_dt_list[0], self.valid_dt_list[-1])
        xticks = self._thin_dtimes_for_xticks(self.valid_dt_list)
        if (self.temporal_res == 24):
            xtick_labels = [f"{xtick:%m/%d}" for xtick in xticks]
        else:
            xtick_labels = [f"{xtick:%m/%d %H}" for xtick in xticks]
        plt.xticks(xticks, xtick_labels, fontsize = 15, rotation = 60)
        if (plot_levels is not None):
            plt.ylim(plot_levels[0], plot_levels[-1])
            plt.yticks(plot_levels, fontsize = 15)
        else:
            plt.ylim(0, 1.0)
            plt.yticks(np.arange(0, 1.1, 0.1), fontsize = 15) 
        plt.grid(True, linewidth = 0.5)
        
        # Plot data
        for data_name in self.data_names:
            if (data_name == self.truth_data_name):
                continue
       
            # If the threshold is equal to the fixed evaluation threshold,
            # data will be within self.fss_dict_by_radius DataArrays. Otherwise, it will
            # be in self.fss_dict_by_threshold DataArrays.
            if (eval_threshold == self.fixed_fss_eval_threshold):
                da = self.fss_dict_by_radius[data_name].loc[:, eval_radius]
            else:
                da = self.fss_dict_by_threshold[data_name].loc[:, eval_threshold]
            plt.plot(da.period_end_time.values, da, color = ppu.datasets_colors_dict[data_name], linewidth = 2.5, label = data_name)  

        # Save figure
        plt.tight_layout()
        plt.legend(loc = "best", prop = {"size": 15})
        fig_name = f"FSStimeseries.{self.data_names_str}radius{eval_radius:0.2f}{self.fss_eval_radius_units}.threshold{eval_threshold:0.1f}{self.fss_eval_threshold_units}.{short_name}.{self.daily_time_period_str}.{self.region}.png"
        fig_path = os.path.join(self.plot_output_dir, fig_name)
        print(f"Saving {fig_path}")
        plt.savefig(fig_path)

    def how_to_OLD_plot_cmap_multi_panel(self):
        print('plot_cmap_multi_panel(data_dict = None, figsize = None, extend = "max",\n'
              '                      plot_levels = self.plot_levels, plot_cmap = self.plot_cmap,\n'
              '                      time_period_type = None, stat_type = "mean", pctl = 99,\n'
              '                      single_colorbar = True, single_set_of_levels = True,\n'
              '                      plot_errors = False, error_plot_levels = None,\n'
              '                      truth_data_contour = None, write_to_nc = False)\n')
        print("NOTE:\n"
              "If data_dict is not None it will be used directly, without aggregation using time_period_type.\n"
              "So in this case, the contents of data_dict must already be properly aggregated for desired contour maps.\n"
              "If data_dict = None and self.USE_EXTERNAL_DA_DICT = False, self.da_dict will be aggregated, then plotted, according to time_period_type and stat_type.\n"
              "If data_dict = None and self.USE_EXTERNAL_DA_DICT = True, self.da_dict will be used directly.\n"
              "So in the latter case, it's better to calculate data_dict with desired aggregation separately and pass it to plot_cmap_multi_panel().")

    # Contour maps with the correct number of panels, with the "truth" dataset always in the top left
    # The input da_dict must have the same contents as self.data_dict: {da_name1: da1, ...., da_nameN, daN}
    # So, best practice is to have whatever data you want to plot exist as concatenated xarray DataArrays
    # with the proper time dimensions, as this method will make a plot for each coordinate of the time dimension.
    def OLD_plot_cmap_multi_panel(self, data_dict = None, figsize = None, extend = "max",
                              plot_levels = None, plot_cmap = None,
                              time_period_type = None, stat_type = "mean", pctl = 99,
                              single_colorbar = True, single_set_of_levels = True,
                              plot_errors = False, error_plot_levels = None,
                              truth_data_contour = None, write_to_nc = False):
        if (data_dict is None):
            if not(self.USE_EXTERNAL_DA_DICT): 
                data_dict = self.calculate_aggregated_stats(time_period_type = time_period_type, 
                                                            stat_type = stat_type,
                                                            agg_type = "time",
                                                            pctl = pctl,
                                                            write_to_nc = write_to_nc) 
            else:
                data_dict = self.da_dict

        if (plot_levels is None):
            plot_levels = self.plot_levels

        if (plot_cmap is None):
            plot_cmap = self.plot_cmap

        # Configure basic info about the data
        truth_da = data_dict[self.truth_data_name]
        num_da = len(data_dict.items())
        data_names_str = "".join(f"{key}." for key in data_dict.keys())

        # Get the contour levels for the plot
        if (plot_levels is not None):
            levels = plot_levels
            _, error_levels = self._calculate_levels_for_cmap(data_dict)
        else: 
            levels, error_levels = self._calculate_levels_for_cmap(data_dict)

        if (error_plot_levels is not None):
            error_levels = error_plot_levels
        
        # Based on this particular dataset, get a list of all the valid datetimes we're going to plot 
        dtimes, time_dim, dt_format = self._create_datetime_list_from_da_time_dim(truth_da)

        # Set map projection to be used for all subplots in the figure 
        proj = ccrs.PlateCarree()

        # Loop through these datetimes, making figures with subplots corresponding to each of the data arrays in data_dict
        for dtime in dtimes:
            if (figsize is not None):
                fig = plt.figure(figsize = figsize)
            else:
                fig = plt.figure(figsize = ppu.set_figsize_based_on_num_da(num_da, self.region_plot_config)) 
            axes_list, cbar_ax = ppu.OLD_create_gridded_subplots(num_da, proj, single_colorbar = single_colorbar) 

            if (type(dtime) is pd.Timestamp):
                loc_str = dtime.strftime(utils.full_date_format_str) # Format is %Y-%m-%d %H:%M:%S
                dt_str = dtime.strftime(dt_format) 
            elif (type(dtime) is str):
                loc_str = dtime
                dt_str = dtime
            else:
                print(f"Invalid datetime {dtime} to select data; not continuing to make plots")
                return 
      
            # Create different contour levels for each dtime rather than a fixed set of levels for all dtimes.
            # Note the levels will all be the same (e.g., error levels) within one single figure, as one output
            # figure corresponds to one dtime. 
            if not(single_set_of_levels):
                data_to_plot_dict = {}
                for data_name, da in data_dict.items():
                    data_to_plot_dict[data_name] = da.loc[loc_str]
                levels, error_levels = self._calculate_levels_for_cmap(data_to_plot_dict)

            # Loop through each of the subplot axes defined above (one axis for each DataArray) and plot the data 
            for axis, (data_name, da) in zip(axes_list, data_dict.items()):
                self._add_cartopy_features_to_map_proj(axis, proj, draw_labels = False)
        
                xy_coords = ppu.determine_xy_coordinates(da)

                if (plot_errors) and (data_name != self.truth_data_name):
                    data_to_plot = (da - truth_da).loc[loc_str]
                    plot_levels = error_levels
                    subplot_title = f"{data_name} errors"
                    cmap = "seismic_r"
                    cbar_extend = "both" # Need to extend colorbar in both directions for error cmaps
                else:
                    data_to_plot = da.loc[loc_str]
                    plot_levels = levels
                    subplot_title = data_name
                    cmap = plot_cmap
                    cbar_extend = extend 
                    if (extend != "min") and (extend != "max") and (extend != "both"):
                        cbar_extend = "both"

                # One colorbar for entire figure; add as its own separate axis defined using subplot2grid 
                if single_colorbar:
                    plot_handle = data_to_plot.plot(ax = axis, levels = plot_levels, transform = proj, extend = cbar_extend, cmap = cmap,
                                                    x = xy_coords.x, y = xy_coords.y, 
                                                    add_colorbar = not(single_colorbar))
                    cbar = fig.colorbar(plot_handle, cax = cbar_ax, ticks = plot_levels, shrink = 0.5, orientation = "horizontal")
                    cbar.set_label(da.units, size = 15)
                    cbar_tick_labels_rotation, cbar_tick_labels_fontsize = ppu.set_cbar_labels_rotation_and_fontsize(self.region_plot_config, plot_levels, num_da, for_single_cbar = True)
                    cbar.ax.set_xticklabels(plot_levels, rotation = cbar_tick_labels_rotation) 
                    cbar.ax.tick_params(labelsize = cbar_tick_labels_fontsize)
                # Separate colorbar for each subplot
                else:
                    plot_handle = data_to_plot.plot(ax = axis, levels = plot_levels, transform = proj, extend = cbar_extend, cmap = cmap,
                                                    x = xy_coords.x, y = xy_coords.y,
                                                    cbar_kwargs = {"shrink": 0.6, "ticks": plot_levels, "pad": 0.02, "orientation": "horizontal"})
                    plot_handle.colorbar.set_label(da.units, size = 15, labelpad = -1.3)
                    cbar_tick_labels = ppu.create_sparse_cbar_ticks(plot_levels) # Label every other tick on subplot colorbars
                    cbar_tick_labels_rotation, cbar_tick_labels_fontsize = \
                    ppu.set_cbar_labels_rotation_and_fontsize(self.region_plot_config, cbar_tick_labels, num_da, for_single_cbar = False)
                    plot_handle.colorbar.ax.set_xticklabels(cbar_tick_labels, rotation = cbar_tick_labels_rotation) 
                    plot_handle.colorbar.ax.tick_params(labelsize = cbar_tick_labels_fontsize)

                # Plot a single, unfilled contour of data from truth_da for visual comparison
                # to the filled contours plotted for the current da.
                if ((type(truth_data_contour) is list) or \
                    (type(truth_data_contour) is np.ndarray)) and \
                   (data_name != self.truth_data_name):
                    truth_da.loc[loc_str].plot.contour(ax = axis, levels = truth_data_contour,
                                                       colors = "black",
                                                       linewidths = 2.5,
                                                       x = xy_coords.x, y = xy_coords.y,)

                axis.set_title(subplot_title, fontsize = 16 + self.poster_font_increase)

            formatted_short_name = pdp.format_short_name(truth_da)
            if plot_errors:
                formatted_short_name += "_errors"
            formatted_short_short_name = self._format_short_short_name(truth_da)

            # Create the plot title. If we're looping through individual Timestamp objects, they represent
            # the period end time of the data. Indicate this explicitly in the title.
            if (type(dtime) is pd.Timestamp):
                title_string = f"{self.region} {formatted_short_name} ending at {dt_str}"
            else:
                title_string = f"{self.region} {formatted_short_name}: {dt_str}"
            fig.suptitle(title_string, fontsize = 16 + self.poster_font_increase, fontweight = "bold")
            fig.tight_layout()

            # Save figure
            if (dt_str in ppu.construct_monthly_string_list()):
                time_period_number = ppu.month_string_to_month_number(dt_str)
                dt_str = f"{time_period_number:02d}{dt_str}.{self.monthly_time_period_str}"
            elif (dt_str in ppu.construct_seasonal_string_list()): 
                time_period_number = ppu.season_string_to_season_number(dt_str)
                dt_str = f"{time_period_number:02d}{dt_str}.{self.monthly_time_period_str}"

            fig_name = f"cmap.{data_names_str}{formatted_short_name}.{dt_str}.{self.region}.png"
            fig_path = os.path.join(self.plot_output_dir, fig_name)
            print(f"Saving {fig_path}")
            plt.savefig(fig_path)

    def how_to_plot_cmap_multi_panel(self):
        print('plot_cmap_multi_panel(data_dict,\n'
              '                      figsize = None,\n'
              '                      subplot_layout = None,\n'
              '                      plot_levels = None,\n'
              '                      plot_color_map = None,\n'
              '                      default_color_map_version = 2,\n'
              '                      extend = "max",\n'
              '                      truth_data_contour = None,\n'
              '                      cbar_tick_labels = None,\n'
              '                      cbar_tick_labels_rotation = None\n'
              '                      cbar_tick_labels_fontsize = None\n'
              '                      sparse_cbar_ticks = False)')

    def plot_cmap_multi_panel(self,
                              data_dict,
                              figsize = None, 
                              subplot_layout = None, 
                              plot_levels = None,
                              plot_color_map = None,
                              default_color_map_version = 2,
                              extend = "max",
                              truth_data_contour = None,
                              cbar_tick_labels = None,
                              cbar_tick_labels_rotation = None,
                              cbar_tick_labels_fontsize = None,
                              sparse_cbar_ticks = False):

        # Configure basic info about the data
        num_da = len(data_dict.items())
        truth_da = data_dict[self.truth_data_name]
        data_names_str = "".join(f"{key}." for key in data_dict.keys())
      
        # Configure subplot layout
        if (subplot_layout is None):
            subplot_layout = self.region_plot_config.subplot_layout
        if (subplot_layout != "grid") and (subplot_layout != "vertical") and (subplot_layout != "horizontal"):
            subplot_layout = "grid"
 
        # Configure plot levels 
        if (plot_levels is None) or (plot_color_map is None):
            plot_color_map, plot_levels, color_list = ppu.create_precip_plot_levels(temporal_res = truth_da.interval_hours,
                                                                                    version = default_color_map_version) 

        # Based on this particular dataset, get a list of all the valid datetimes we're going to plot 
        dtimes, time_dim, dt_format = self._create_datetime_list_from_da_time_dim(truth_da)

        # Set map projection to be used for all subplots in the figure 
        proj = ccrs.PlateCarree()
        
        # Loop through these datetimes, making figures with subplots corresponding to each of the data arrays in data_dict
        for dtime in dtimes:
            # Set figure size
            # Constrained layout hasn't worked well (color bar overlaps with plots), but consider experimenting in future
            if (figsize is None):
                figsize = self.region_plot_config.figsize
            fig = plt.figure(figsize = figsize) #, layout = "constrained")

            # Configure subplot axes
            axes = ppu.create_gridded_subplots(num_da, proj = proj, layout = subplot_layout) 

            # Configure correct date/time string to use for indexing the data
            if (type(dtime) is pd.Timestamp):
                loc_str = dtime.strftime(utils.full_date_format_str) # Format is %Y-%m-%d %H:%M:%S
                dt_str = dtime.strftime(dt_format) 
            elif (type(dtime) is str):
                loc_str = dtime
                dt_str = dtime
            else:
                print(f"Error: Invalid datetime {dtime} to select data; not continuing to make plots")
                return 

            # Loop through each of the subplot axes defined above (one axis for each DataArray) and plot the data 
            for axis, (data_name, da) in zip(axes, data_dict.items()):
                self._add_cartopy_features_to_map_proj(axis, proj, draw_labels = False)
                xy_coords = ppu.determine_xy_coordinates(da)

                # Plot the data as filled contours
                if (extend != "min") and (extend != "max") and (extend != "both"):
                    extend = "both"
                contour_plot = da.loc[loc_str].plot(ax = axis,
                                                    transform = proj,
                                                    x = xy_coords.x,
                                                    y = xy_coords.y, 
                                                    levels = plot_levels,
                                                    cmap = plot_color_map,
                                                    extend = extend, 
                                                    add_colorbar = False) 
    
                # Plot a single, unfilled contour of data from truth_da for visual comparison
                # to the filled contours plotted for the current da.
                if isinstance(truth_data_contour, (int, float)) and (data_name != self.truth_data_name):
                    truth_da.loc[loc_str].plot.contour(ax = axis,
                                                       x = xy_coords.x,
                                                       y = xy_coords.y,
                                                       levels = [truth_data_contour],
                                                       colors = "black",
                                                       linewidths = 2.5)

                # Set subplot title AFTER plotting to override what xarray will put there using the .plot wrapper
                axis.set_title(data_name, fontsize = 15 + self.poster_font_increase)
          
            # Configure colorbar ticks and labels
            if cbar_tick_labels is None:
                if sparse_cbar_ticks:
                    final_cbar_tick_labels = ppu.create_sparse_cbar_ticks(plot_levels) # Label every other tick on subplot colorbars
                else:
                    final_cbar_tick_labels = ppu.set_plot_levels_whole_numbers_to_integers(plot_levels)
            else:
                final_cbar_tick_labels = ppu.set_plot_levels_whole_numbers_to_integers(cbar_tick_labels)

            rotation, fontsize = ppu.set_cbar_labels_rotation_and_fontsize(self.region_plot_config, plot_levels, num_da, for_single_cbar = True)
            if (cbar_tick_labels_rotation is None):
                cbar_tick_labels_rotation = rotation
            if (cbar_tick_labels_fontsize is None):
                cbar_tick_labels_fontsize = fontsize 

            # Colorbars with extend = "max" are very annoyingly off-center, too far to the left.
            # So, move them to the right a bit via axes positioning (items in list passed to 
            # add_axes method are [left, bottom, width, height]
            if (extend == "max"):
                cbar_ax = fig.add_axes([0.275, 0.1, 0.5, 0.02])
            else:
                cbar_ax = fig.add_axes([0.25, 0.1, 0.5, 0.02])
            cbar = fig.colorbar(contour_plot, cax = cbar_ax, orientation = "horizontal", ticks = plot_levels)
            cbar.set_label(truth_da.units, size = 14 + self.poster_font_increase)
            cbar.ax.set_xticklabels(final_cbar_tick_labels, rotation = cbar_tick_labels_rotation) 
            cbar.ax.tick_params(labelsize = cbar_tick_labels_fontsize) 

            # Adjust subplots position to make room for colorbar
            fig.subplots_adjust(bottom = 0.15)

            # Create the plot title.
            # If looping through Timestamp objects, they represent period end times. Indicate this explicitly in the title.
            formatted_short_name = pdp.format_short_name(truth_da)
            if (type(dtime) is pd.Timestamp):
                title_string = f"{self.region} {formatted_short_name} ending at {dt_str}"
            else:
                title_string = f"{self.region} {formatted_short_name}: {dt_str}"
            fig.suptitle(title_string, fontsize = 16 + self.poster_font_increase, fontweight = "bold", y = 0.95)

            # Save figure
            if (dt_str in ppu.construct_monthly_string_list()):
                time_period_number = ppu.month_string_to_month_number(dt_str)
                dt_str = f"{time_period_number:02d}{dt_str}.{self.monthly_time_period_str}"
            elif (dt_str in ppu.construct_seasonal_string_list()): 
                time_period_number = ppu.season_string_to_season_number(dt_str)
                dt_str = f"{time_period_number:02d}{dt_str}.{self.monthly_time_period_str}"

            fig_name = f"cmap.{data_names_str}{formatted_short_name}.{dt_str}.{self.region}.png"
            fig_path = os.path.join(self.plot_output_dir, fig_name)
            print(f"Saving {fig_path}")
            plt.savefig(fig_path, bbox_inches = "tight")
    
    def how_to_plot_cmap_multi_panel_errors(self):
        print('plot_cmap_multi_panel_errors(data_dict,\n'
              '                             figsize = None,\n'
              '                             subplot_layout = None,\n'
              '                             plot_levels = None,\n'
              '                             error_levels = None,\n'
              '                             plot_color_map = None,\n'
              '                             default_color_map_version = 2,\n'
              '                             extend = "max",\n'
              '                             cbar_tick_labels = None,\n'
              '                             cbar_tick_labels_rotation = None\n'
              '                             cbar_tick_labels_fontsize = None\n'
              '                             sparse_cbar_ticks = False)')

    def plot_cmap_multi_panel_errors(self,
                                     data_dict,
                                     figsize = None, 
                                     subplot_layout = None, 
                                     plot_levels = None,
                                     error_levels = None,
                                     plot_color_map = None,
                                     default_color_map_version = 2,
                                     extend = "max",
                                     cbar_tick_labels = None,
                                     cbar_tick_labels_rotation = None,
                                     cbar_tick_labels_fontsize = None,
                                     sparse_cbar_ticks = False):

        # Configure basic info about the data
        num_da = len(data_dict.items())
        truth_da = data_dict[self.truth_data_name]
        data_names_str = "".join(f"{key}." for key in data_dict.keys())
        
        # Configure subplot layout
        if (subplot_layout is None):
            subplot_layout = self.region_plot_config.subplot_layout
        if (subplot_layout != "grid") and (subplot_layout != "vertical") and (subplot_layout != "horizontal"):
            subplot_layout = "grid"
       
        # Configure plot levels 
        if (plot_levels is None) or (plot_color_map is None):
            plot_color_map, plot_levels, color_list = ppu.create_precip_plot_levels(temporal_res = truth_da.interval_hours,
                                                                                    version = default_color_map_version) 

        if (error_levels is None):
            _, error_levels = self._calculate_levels_for_cmap(data_dict)
    
        # Based on this particular dataset, get a list of all the valid datetimes we're going to plot 
        dtimes, time_dim, dt_format = self._create_datetime_list_from_da_time_dim(truth_da)

        # Set map projection to be used for all subplots in the figure 
        proj = ccrs.PlateCarree()
        
        # Loop through these datetimes, making figures with subplots corresponding to each of the data arrays in data_dict
        for dtime in dtimes:
            # Set figure size
            if (figsize is None):
                figsize = self.region_plot_config.figsize_errors
            fig = plt.figure(figsize = figsize)

            # Configure subplot axes
            axes = ppu.create_gridded_subplots(num_da, proj = proj, layout = subplot_layout) 

            # Configure correct date/time string to use for indexing the data
            if (type(dtime) is pd.Timestamp):
                loc_str = dtime.strftime(utils.full_date_format_str) # Format is %Y-%m-%d %H:%M:%S
                dt_str = dtime.strftime(dt_format) 
            elif (type(dtime) is str):
                loc_str = dtime
                dt_str = dtime
            else:
                print(f"Error: Invalid datetime {dtime} to select data; not continuing to make plots")
                return 

            # Loop through each of the subplot axes defined above (one axis for each DataArray) and plot the data 
            for axis, (data_name, da) in zip(axes, data_dict.items()):
                self._add_cartopy_features_to_map_proj(axis, proj, draw_labels = False)
                xy_coords = ppu.determine_xy_coordinates(da)

                # Plot the data as filled contours
                if (data_name != self.truth_data_name):
                    data_to_plot = (da - truth_da).loc[loc_str]
                    final_plot_levels = error_levels
                    final_plot_color_map = "seismic_r"
                    subplot_title = f"{data_name} errors"
                    cbar_extend = "both" # Need to extend colorbar in both directions for error cmaps
                    final_cbar_tick_labels = ppu.create_sparse_cbar_ticks(error_levels) # Label every other tick on subplot colorbars 
                else:
                    data_to_plot = da.loc[loc_str]
                    final_plot_levels = plot_levels 
                    final_plot_color_map = plot_color_map
                    subplot_title = data_name

                    if (extend != "min") and (extend != "max") and (extend != "both"):
                        cbar_extend = "both"
                    else:
                        cbar_extend = extend
            
                    # Configure colorbar ticks and labels
                    if cbar_tick_labels is None:
                        if sparse_cbar_ticks:
                            final_cbar_tick_labels = ppu.create_sparse_cbar_ticks(plot_levels) # Label every other tick on subplot colorbars
                        else:
                            final_cbar_tick_labels = ppu.set_plot_levels_whole_numbers_to_integers(plot_levels)
                    else:
                        final_cbar_tick_labels = ppu.set_plot_levels_whole_numbers_to_integers(cbar_tick_labels)

                contour_plot = data_to_plot.plot(ax = axis,
                                                 transform = proj,
                                                 x = xy_coords.x,
                                                 y = xy_coords.y, 
                                                 levels = final_plot_levels,
                                                 cmap = final_plot_color_map,
                                                 extend = cbar_extend, 
                                                 cbar_kwargs = {"orientation": "horizontal",
                                                                "shrink": 0.6,
                                                                "aspect": 60, 
                                                                "pad": 0.02,
                                                                "ticks": final_plot_levels})

                # Configure colorbar
                rotation, fontsize = ppu.set_cbar_labels_rotation_and_fontsize(self.region_plot_config, final_plot_levels, num_da, for_single_cbar = False)
                if (cbar_tick_labels_rotation is None):
                    cbar_tick_labels_rotation = rotation
                if (cbar_tick_labels_fontsize is None):
                    cbar_tick_labels_fontsize = fontsize

                #contour_plot.colorbar.set_label(da.units, size = 14, labelpad = -35, x = 1.15)
                contour_plot.colorbar.set_label("")
                contour_plot.colorbar.ax.set_xticklabels(final_cbar_tick_labels, rotation = cbar_tick_labels_rotation) 
                contour_plot.colorbar.ax.tick_params(labelsize = cbar_tick_labels_fontsize)

                # Annotate units near right edge of colorbar 
                axis.annotate(da.units,
                              xy = (0.81, -0.17), 
                              xytext = (0.81, -0.17), 
                              xycoords = "axes fraction",
                              size = 14) 
    
                # Set subplot title AFTER plotting to override what xarray will put there using the .plot wrapper
                axis.set_title(subplot_title, fontsize = 15 + self.poster_font_increase)
          
            # Create the plot title.
            # If looping through Timestamp objects, they represent period end times. Indicate this explicitly in the title.
            formatted_short_name = pdp.format_short_name(truth_da) + "_errors"
            if (type(dtime) is pd.Timestamp):
                title_string = f"{self.region} {formatted_short_name} ending at {dt_str}"
            else:
                title_string = f"{self.region} {formatted_short_name}: {dt_str}"
            fig.suptitle(title_string, fontsize = 16 + self.poster_font_increase, fontweight = "bold", y = 0.95)

            # Save figure
            if (dt_str in ppu.construct_monthly_string_list()):
                time_period_number = ppu.month_string_to_month_number(dt_str)
                dt_str = f"{time_period_number:02d}{dt_str}.{self.monthly_time_period_str}"
            elif (dt_str in ppu.construct_seasonal_string_list()): 
                time_period_number = ppu.season_string_to_season_number(dt_str)
                dt_str = f"{time_period_number:02d}{dt_str}.{self.monthly_time_period_str}"

            fig_name = f"cmap.{data_names_str}{formatted_short_name}.{dt_str}.{self.region}.png"
            fig_path = os.path.join(self.plot_output_dir, fig_name)
            print(f"Saving {fig_path}")
            plt.savefig(fig_path, bbox_inches = "tight")

    def how_to_plot_timeseries(self):
        print('plot_timeseries(data_dict = None, time_period_type = None, stat_type = "mean", pctl = 99, plot_levels = None, \n'
              '                calc_stats = True, write_stats = True, ann_plot = True, which_ann_text = "all", pct_errors_ann_text = True,\n'
              '                plot_truth_data = True, plot_zero_line = False, write_to_nc = False, full_short_name_in_fig_name = False)')
        print("NOTE:\n"
              "If data_dict is not None it will be used directly, without aggregation using time_period_type.\n"
              "So the contents of data_dict must already be properly aggregated for a time series.\n"
              "If data_dict = None and self.USE_EXTERNAL_DA_DICT = False, self.da_dict will be aggregated, then plotted, according to time_period_type and stat_type.\n"
              "If data_dict = None and self.USE_EXTERNAL_DA_DICT = True, self.da_dict will be used directly.\n"
              "So in the latter case, it's better to calculate data_dict with desired aggregation separately and pass it to plot_timeseries().")

    def plot_timeseries(self, data_dict = None, time_period_type = None, stat_type = "mean", pctl = 99, plot_levels = None,
                        calc_stats = True, write_stats = True, ann_plot = True, which_ann_text = "all", pct_errors_ann_text = True,
                        plot_truth_data = True, plot_zero_line = False, write_to_nc = False, full_short_name_in_fig_name = False):
        if (data_dict is None):
            if not(self.USE_EXTERNAL_DA_DICT):
                data_dict = self.calculate_aggregated_stats(time_period_type = time_period_type, 
                                                            stat_type = stat_type,
                                                            agg_type = "space_time",
                                                            pctl = pctl,
                                                            write_to_nc = write_to_nc)
            else:
                data_dict = self.da_dict

        truth_da = data_dict[self.truth_data_name] 
        num_da = len(data_dict.items())
        yticks = plot_levels

        match stat_type:
            case "pctl":
                title_string = f"{pctl:0.1f}th {stat_type}, {self.region}"
                fig_name_prefix = f"{pctl:0.1f}th_{stat_type}"
                if (plot_levels is None):
                    yticks = ppu.variable_pctl_plot_limits("accum_precip", self.temporal_res)
            case "pctl_exclude_zeros":
                title_string = f"{pctl:0.1f}th {stat_type}, {self.region}"
                fig_name_prefix = f"{pctl:0.1f}th_{stat_type}"
                if (plot_levels is None):
                    yticks = ppu.variable_pctl_plot_limits("accum_precip", self.temporal_res)
            case "mean": 
                title_string = f"{stat_type.title()}, {self.region}"
                fig_name_prefix = stat_type 
                if (plot_levels is None):
                    yticks = self.region_plot_config.ts_mean_precip_range 
                    if (time_period_type == None) or (time_period_type == "daily"):
                        yticks = 2.0 * np.copy(yticks)
            case "mean_exclude_zeros": 
                title_string = f"{stat_type.title()}, {self.region}"
                fig_name_prefix = stat_type
                if (plot_levels is None):
                    yticks = self.region_plot_config.ts_mean_precip_range 
                    if (time_period_type == None) or (time_period_type == "daily"):
                        yticks = 2.0 * np.copy(yticks)
            case "max": 
                title_string = f"{stat_type.title()}, {self.region}"
                fig_name_prefix = stat_type
                if (plot_levels is None):
                    yticks = ppu.variable_plot_limits("accum_precip", temporal_res = self.temporal_res)
            case "pctl_excd_mean":
                title_string = f"{pctl:0.1f}th pctl exceedance mean, {self.region}"
                fig_name_prefix = f"{pctl:0.1f}th_{stat_type}"
                if (plot_levels is None):
                    yticks = ppu.variable_pctl_plot_limits("accum_precip", self.temporal_res)
            case "count":
                title_string = truth_da.long_name 
                fig_name_prefix = pdp.format_short_name(truth_da, make_lower = True) 
                if (plot_levels is None):
                    yticks, error_levels = self._calculate_levels_for_cmap(data_dict, mean_of_max = False, saturate = False) 
            case _:
                title_string = f"{stat_type.title()}, {self.region}"
                fig_name_prefix = stat_type.lower().replace(' ', '_').replace('-', '_') 
                if (plot_levels is None):
                    yticks = ppu.variable_plot_limits("accum_precip", temporal_res = self.temporal_res)
                     
        if (time_period_type is not None):
            fig_name_prefix = f"{time_period_type}_{fig_name_prefix}"

        match time_period_type:
            case None:
                time_period_str = self.daily_time_period_str
                title_string = f"{title_string}: {time_period_str}"
                xlabel = "Period end time"
            case "daily":
                time_period_str = self.daily_time_period_str_period_begin 
                title_string = f"Daily {title_string}: {time_period_str}"
                xlabel = "Days"
            case "monthly":
                time_period_str = self.monthly_time_period_str
                title_string = f"Monthly {title_string}: {time_period_str}"
                xlabel = "Months"
            case "seasonal":
                time_period_str = self.monthly_time_period_str
                title_string = f"Seasonal {title_string}: {time_period_str}"
                xlabel = "Seasons"
            case "annual":
                time_period_str = self.annual_time_period_str
                title_string = f"Annual {title_string}: {time_period_str}"
                xlabel = "Years"
            case "common_monthly":
                time_period_str = self.monthly_time_period_str
                title_string = f"Common monthly {title_string}: {time_period_str}"
                xlabel = "Months"
            case "common_seasonal": 
                time_period_str = self.monthly_time_period_str
                title_string = f"Common seasonal {title_string}: {time_period_str}"
                xlabel = "Seasons"

        formatted_short_name = pdp.format_short_name(truth_da)
        formatted_short_short_name = self._format_short_short_name(truth_da)
        if (stat_type == "count"):
            fig_name = f"timeseries.{self.data_names_str}{fig_name_prefix}.{time_period_str}.{self.region}.png"
        else:
            if full_short_name_in_fig_name:
                fig_name = f"timeseries.{self.data_names_str}{fig_name_prefix}.{formatted_short_name}.{time_period_str}.{self.region}.png"
            else:
                fig_name = f"timeseries.{self.data_names_str}{fig_name_prefix}.{formatted_short_short_name}.{time_period_str}.{self.region}.png"
        dtimes, time_dim, dt_format = self._create_datetime_list_from_da_time_dim(truth_da)
       
        # Plot the data
        fig = plt.figure(figsize = (15, 10))
        for data_name, data_array in data_dict.items():
            if (data_name == self.truth_data_name) and not(plot_truth_data):
                continue
            color = ppu.datasets_colors_dict[data_name]
            data_array.plot(label = data_name, color = color, linewidth = 2.5, linestyle = "solid", zorder = 1)
        if plot_zero_line:
            plt.plot([dtimes[0], dtimes[-1]], [0, 0], linewidth = 3, color = "black", zorder = 0) 
        plt.legend(loc = "upper right", prop = {"size": 15 + self.poster_font_increase})

        # Configure plot title and axes labels 
        plt.grid(True, linewidth = 0.3, linestyle = "dashed")
        plt.title(title_string, size = 16 + self.poster_font_increase, fontweight = "bold")
        plt.xlabel(xlabel, size = 16)
        if (stat_type == "count"): 
            plt.ylabel(f"{truth_da.units}", size = 16 + self.poster_font_increase)
        else:
            plt.ylabel(f"{formatted_short_name} [{truth_da.units}]", size = 16 + self.poster_font_increase)

        # Set axes ticks, labels, and limits
        xticks = self._thin_dtimes_for_xticks(dtimes)
        if (dt_format != ""):
            if (time_period_type == None):
                if (self.temporal_res == 24):
                    dt_format = "%m/%d"
                else:
                    dt_format = "%m/%d %H"
            xtick_labels = [xtick.strftime(dt_format) for xtick in xticks]
        else:
            xtick_labels = [xtick for xtick in xticks]
        plt.xticks(xticks, xtick_labels, fontsize = 15 + self.poster_font_increase, rotation = 60)
        plt.yticks(yticks, fontsize = 15 + self.poster_font_increase)
        plt.xlim(dtimes[0], dtimes[-1])
        plt.ylim(yticks[0], yticks[-1])
        plt.tight_layout()

        # Calculate basic statistics and annotate on the plot
        # Only calculate these statistics if it makes sense to (plot_truth_data_name = True,
        # so there is a truth timeseries against which to calculate stats) and calc_stats = True.
        if (plot_truth_data and calc_stats): 
            print("Calculating statistics to annotate timeseries plot")
            if (num_da < 3):
                ann_pos_step = 0.40
                ann_size = 15
            elif (num_da == 3):
                ann_pos_step = 0.35
                ann_size = 15
            elif (num_da == 4):
                ann_pos_step = 0.25
                ann_size = 14
            elif (num_da == 5):
                ann_pos_step = 0.20
                ann_size = 12
            else:
                ann_pos_step = 0.17
                ann_size = 10
            
            ann_pos_horz = 0.01
            ann_pos_vert = 0.86
            range_truth_data = truth_da.max().load().item() - truth_da.min().load().item()
            ann_text_dict = {}
            for data_name, da in data_dict.items():
                print(f"****** {data_name}")
                if (data_name == self.truth_data_name):
                    continue

                # Calculate stats
                correlation = self.calculate_pearsonr(da, truth_da)
                errors = (da - truth_da).load()
                squared_errors = errors**2
                mean_bias = errors.mean().item()
                norm_mean_bias = 100 * mean_bias/range_truth_data
                rmse = np.sqrt(squared_errors.mean().item())
                norm_rmse = 100 * rmse/range_truth_data

                # Create the text
                data_name_text = f"{data_name}:\n"
                corr_text = f"Corr: {correlation.statistic:0.2f} (p-value: {correlation.pvalue:0.2f})\n"
                bias_text = f"Mean bias: {mean_bias:0.2f}{truth_da.units}\n"
                rmse_text = f"RMSE: {rmse:0.2f}{truth_da.units}\n"

                if pct_errors_ann_text:
                    bias_text = bias_text.strip("\n") + f" ({norm_mean_bias:0.1f}%)\n"
                    rmse_text = rmse_text.strip("\n") + f" ({norm_rmse:0.1f}%)\n"

                if (which_ann_text == "corr"):
                    ann_text = data_name_text + corr_text
                elif (which_ann_text == "errors"):
                    ann_text = data_name_text + bias_text + rmse_text
                else: 
                    ann_text = data_name_text + corr_text + bias_text + rmse_text

                # Annotate the plot
                print(ann_text)
                ann_text_dict[data_name] = ann_text
                if ann_plot:
                    plt.annotate(ann_text, weight = "bold",
                                 xy = (ann_pos_horz, ann_pos_vert),
                                 xytext = (ann_pos_horz, ann_pos_vert),
                                 xycoords = "axes fraction", size = ann_size)
                ann_pos_horz += ann_pos_step

                if write_stats:
                    stats_dir = os.path.join(self.output_dir, "stats")
                    stats_fname = fig_name.split(".png")[0] + ".txt"
                    stats_fpath = os.path.join(stats_dir, stats_fname)
                    print(f"Writing stats to {stats_fpath}")
                    Fstats = open(stats_fpath, "w")

                    for data_name, ann_text in ann_text_dict.items():
                        Fstats.write(ann_text)
                    Fstats.close()

        # Save figure
        fig_fpath = os.path.join(self.plot_output_dir, fig_name)
        print(f"Saving {fig_fpath}")
        plt.savefig(fig_fpath)

    # TODO: Consider plotting at midpoint of bin rather than left edge of bin
    def plot_pdf(self, data_dict = None, time_period_type = "full_period", write_to_nc = False, bins = 10):
        if (data_dict is None):
            if not(self.USE_EXTERNAL_DA_DICT):
                data_dict = self.calculate_pdf(time_period_type = time_period_type, bins = bins) 
            else:
                data_dict = self.da_dict

        # Plot data
        for dtime in data_dict.keys():
            da_dict_this_dtime = data_dict[dtime]
            data_names_str = "".join(f"{key}." for key in da_dict_this_dtime.keys())

            # Calculate reasonable axis ticks and bounds 
            # Compile lists of maximum bins and minimum probabilities
            max_bin_list = []
            min_prob_list = []
            max_total_samples_list = []
            for data_name, (pdf_values, pdf_bins, total_samples) in da_dict_this_dtime.items():
                pdf_bin_max = np.max(pdf_bins) 
                max_bin_list.append(pdf_bin_max)
                prob_min = np.min(pdf_values[pdf_values > 0.0])
                min_prob_list.append(prob_min)
                max_total_samples_list.append(total_samples)
            
            # Calculate x-axis ticks based on overall max bin
            if (type(bins) is list) or (type(bins) is np.ndarray):
                xticks = bins[:-1]
                xlabels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(xticks))]
            else:
                overall_max = np.max(np.array(max_bin_list)) 
                rounded_overall_max = np.round(overall_max, decimals = -1) + 10
                step_size = np.round(rounded_overall_max/10., decimals = -1) 
                xticks = np.arange(0, rounded_overall_max + step_size, step_size)
                xlabels = None

            # Calculate y-axis (log scale) ticks based on overall min probability
            overall_min = np.min(np.array(min_prob_list))
            exp_overall_min = np.floor(np.log10(overall_min))
            yticks = [10**i for i in np.arange(exp_overall_min, 1, 1)] 

            # Initialize figure and plot data 
            plt.figure(figsize = (10,10))
            for data_name, (pdf_values, pdf_bins, total_samples) in da_dict_this_dtime.items():
                plt.plot(pdf_bins[:-1], pdf_values, label = data_name, color = ppu.datasets_colors_dict[data_name], linewidth = 2.5)
            plt.gca().set_yscale("log")
            plt.grid(True, linewidth = 0.5)
            plt.legend(loc = "upper right", prop = {"size": 15})

            # Configure plot title and figure name
            short_name = self.da_dict[self.truth_data_name].short_name
            units = self.da_dict[self.truth_data_name].units
            if (type(dtime) is pd.Timestamp):
                dt_str = dtime.strftime("%Y%m") 
            elif (type(dtime) is str):
                dt_str = dtime
            plt.xlabel(f"{short_name} [{units}]", size = 16)
            plt.ylabel("Probability", size = 16)
            plt.xlim(xticks[0], xticks[-1])
            plt.xticks(xticks, fontsize = 15 + self.poster_font_increase)
            if xlabels is not None:
                plt.gca().set_xticklabels(xlabels, rotation = 45)
            plt.ylim(yticks[0], yticks[-1])
            plt.yticks(yticks, fontsize = 15 + self.poster_font_increase)
            title_string = f"{pdp.format_short_name(self.truth_da)} PDF, {self.region}: {dt_str}"
            plt.title(title_string, size = 16 + self.poster_font_increase, fontweight = "bold")
            plt.tight_layout()

            # Save figure
            if (dt_str in ppu.construct_monthly_string_list()):
                time_period_number = ppu.month_string_to_month_number(dt_str)
                dt_str = f"{time_period_number:02d}{dt_str}.{self.monthly_time_period_str}"
            elif (dt_str in ppu.construct_seasonal_string_list()): 
                time_period_number = ppu.season_string_to_season_number(dt_str)
                dt_str = f"{time_period_number:02d}{dt_str}.{self.monthly_time_period_str}"

            fig_name = f"pdf.{data_names_str}{time_period_type}.{dt_str}.{self.region}.png"
            fig_fpath = os.path.join(self.plot_output_dir, fig_name)
            print(f"Saving {fig_fpath}")
            plt.savefig(fig_fpath)

    # Basic scatter plot of truth dataset versus other dataset 
    def plot_scatter_plot(self, data_dict = None, summed_over_time_period = False):
        if (data_dict is None):
            data_dict = self.da_dict
            truth_da = self.truth_da

        if summed_over_time_period: 
            self.sum_data_over_full_time_period()
            self.truth_da_summed_time = self.da_dict_summed_time[self.truth_data_name]
            data_dict = self.da_dict_summed_time
            truth_da = self.truth_da_summed_time
        
        for data_name, da in data_dict.items(): 
            if (data_name == self.truth_data_name):
                continue

            # Plot data
            plt.figure(figsize = (10, 10))
            current_axes = pylab.gca()
            plt.scatter(da, truth_da) 
            plt.xlabel(f"{data_name} {da.short_name} [{da.units}]", size = 15)
            plt.ylabel(f"{self.truth_data_name} {self.truth_da.short_name} [{self.truth_da.units}]", size = 15)

            # Annotate with correlation
            correlation = self.calculate_pearsonr(da, truth_da)
            ann_text = f"Corr: {correlation.statistic:0.2f} (p-value: {correlation.pvalue:0.2f})"
            plt.annotate(ann_text, xy = (0.05, 0.95), xytext = (0.05, 0.95), xycoords = "axes fraction", size = 12)

            # Format plot and save figure
            plt.title(f"{self.truth_data_name} vs. {data_name} scatter plot: {da.short_name}, {self.daily_time_period_str}", size = 15)
            plt.grid(True, linewidth = 0.5, linestyle = "dashed")
            axes_labels, _ = self._calculate_levels_for_cmap({data_name: da, self.truth_data_name: self.truth_da})
            plot_lims = [axes_labels[0], axes_labels[-1]] 
            plt.xlim(plot_lims)
            plt.ylim(plot_lims)
            plt.plot(plot_lims, plot_lims, color = "black", linewidth = 2.5)
            fig_name = f"scatter_plot.{self.truth_data_name}.{data_name}.{pdp.format_short_name(da)}.{self.daily_time_period_str}.{self.region}.png"
            fig_path = os.path.join(self.plot_output_dir, fig_name)
            print(f"Saving {fig_path}")
            plt.savefig(fig_path)
    ##### END Public methods plotting #####

    ##### Private methods to support plotting #####
    def _add_cartopy_features_to_map_proj(self, axis, data_proj, draw_labels = False):
            axis.coastlines()
            axis.set_extent(self.region_plot_config.region_extent, crs = data_proj)
            axis.add_feature(cfeature.BORDERS)
            if ("US" in self.region):
                axis.add_feature(cfeature.STATES)
            gl = axis.gridlines(crs = data_proj, color = "gray", alpha = 0.5, draw_labels = draw_labels,
                                linewidth = 0.5, linestyle = "dashed")

    # Calculate plot levels for a filled contour map
    def _calculate_levels_for_cmap(self, da_dict, mean_of_max = True, saturate = True):
        max_list = []
        for data_name, da in da_dict.items():
            da_max = da.max().item()
            max_list.append(xr.DataArray(da_max))
        max_da = xr.concat(max_list, dim = "x")

        # Changed from taking the max of the max of each data array to the
        # mean to add more "pop"/saturation to the contour levels, especially for very high pctls (e.g. 99.9th).
        # This approach attempts to address the issues where one dataset has a much higher max than the rest,
        # which can result in washing out contrast in the contour plots. 
        if mean_of_max:
            overall_max = max_da.mean().item()
        else:
            overall_max = max_da.max().item()

        if (overall_max <= 25):
            step_size = 1
            error_levels = np.arange(-5, 5.5, 0.5)
        elif (overall_max <= 50):
            step_size = 2
            error_levels = np.arange(-5, 5.5, 0.5)
        elif (overall_max <= 75):
            step_size = 4
            error_levels = np.arange(-10, 11, 1)
        elif (overall_max <= 100):
            step_size = 5
            error_levels = np.arange(-10, 11, 1)
        elif (overall_max <= 180):
            step_size = 5
            error_levels = np.arange(-20, 22, 2)
        elif (overall_max <= 300):
            step_size = 10
            error_levels = np.arange(-40, 45, 5) 
        elif (overall_max <= 800):
            step_size = 20
            error_levels = np.arange(-140, 150, 10)
        elif (overall_max <= 2000):
            step_size = 50
            error_levels = np.arange(-300, 325, 25)
        elif (overall_max <= 10000):
            step_size = 500
            error_levels = np.arange(-300, 325, 25)
        elif (overall_max <= 100000):
            step_size = 2000
            error_levels = np.arange(-300, 325, 25)
        else:
            step_size = 5000
            error_levels = np.arange(-600, 650, 50)

        overall_max = int(np.round(overall_max, decimals = 0) + step_size)
        overall_max -= overall_max % step_size
        
        # The idea of saturate is to make the plots 'pop' and really highlight
        # the areas of maximum precip. Otherwise if saturate=False, attempt
        # to capture the entire dynamic range of precipitation (the downside in that
        # case is that areas of low precip become very hard to differentiate). 
        if saturate:
            levels = np.arange(0, overall_max - 3 * step_size, step_size)
        else:
            levels = np.arange(0, overall_max + step_size, step_size)
       
        return levels, error_levels

    # Create a list of datetimes (which may be actual datetime objects
    # or strings (e.g., season strings like 'DJF') depending on the 
    # dimension name of the input data array.
    def _create_datetime_list_from_da_time_dim(self, da):
        dims = da.dims
        create_dt_list = True

        if (utils.time_dim_str in dims):
            time_dim = utils.time_dim_str
            dt_format = "%Y%m%d.%H%M" 
        elif (utils.days_dim_str in dims):
            time_dim = utils.days_dim_str
            dt_format = "%Y%m%d"
        elif (utils.period_begin_time_dim_str in dims):
            time_dim = utils.period_begin_time_dim_str 
            dt_format = "%Y%m%d.%H" 
        elif (utils.period_end_time_dim_str in dims): 
            time_dim = utils.period_end_time_dim_str
            dt_format = "%Y%m%d.%H" 
        elif (utils.months_dim_str in dims):
            time_dim = utils.months_dim_str
            dt_format = "%Y%m"
        elif (utils.seasons_dim_str in dims):
            time_dim = utils.seasons_dim_str
            dt_format = ""
            create_dt_list = False
            dtimes = list(da[utils.seasons_dim_str].values)
        elif (utils.annual_dim_str in dims):
            time_dim = utils.annual_dim_str
            dt_format = "%Y"
        elif (utils.full_period_dim_str in dims):
            time_dim = utils.full_period_dim_str
            dt_format = ""
            create_dt_list = False
            dtimes = [ str(da[time_dim].values[0]) ] 
        elif (utils.common_month_dim_str in dims):
            time_dim = utils.common_month_dim_str
            dt_format = ""
            create_dt_list = False
            dtimes = ppu.construct_monthly_string_list()
        elif (utils.common_season_dim_str in dims):
            time_dim = utils.common_season_dim_str
            dt_format = "" 
            create_dt_list = False
            dtimes = ppu.construct_seasonal_string_list()
        else:
            print(f"Error: time dimension not recognized from dimension list {dims}; not making monthly timeseries plot")
            return
        
        if create_dt_list:
            dtimes = [pd.Timestamp(i) for i in da[time_dim].values]
 
        return dtimes, time_dim, dt_format

    # Further format a DataArray's short name from the format separated
    # by underscores to only the first three items separated by underscores.
    def _format_short_short_name(self, data_array):
        formatted_short_name = pdp.format_short_name(data_array)
        str_split = formatted_short_name.split("_")
        formatted_short_short_name = f"{str_split[0]}_{str_split[1]}_{str_split[2]}" 
        return formatted_short_short_name

    def _thin_dtimes_for_xticks(self, dtimes):
        if (len(dtimes) <= 60):
            return dtimes
        elif (len(dtimes) <= 120):
            return dtimes[::2]
        elif (len(dtimes) <= 240):
            return dtimes[::4]
        else:
            return dtimes[::6]
    ##### END Private methods to support plotting #####

