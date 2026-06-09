import cartopy.crs as ccrs
import cartopy.feature as cfeature
import dataclasses
import precip_data_processors
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import regionmask
import sys
import utilities as utils

DEFAULT_PRECIP_CMAP = "terrain_r"

@dataclasses.dataclass
class RegionPlottingConfiguration:
    region_extent: [int, int, int, int]
    figsize_sp: (int, int) # Default figure size for single-panel plots
    figsize: (int, int) # Default figure size for multi-panel plots 
    figsize_errors: (int, int) # Default figure size for multi-panel error plots 
    subplot_layout: str # Layout of subplots for multi-panel plot ('grid', 'vertical', or 'horizontal')
    central_point: [int, int] # Qualitative central point (for annotating text, etc.)
    crosses_meridian: True
    cm_mean_precip_range: np.ndarray # Contour map precip color map range
    ts_mean_precip_range: np.ndarray # Time series precip plot limits range

def print_region_config_info(region):
    region_config = regions_info_dict[region]
    print(f"Region is: {region}\n"
          f"Region extent: {region_config.region_extent}\n"
          f"Default figure size single panel: {region_config.figsize_sp}\n"
          f"Default figure size multi panel: {region_config.figsize}\n"
          f"Default figure size multi panel error plots: {region_config.figsize_errors}\n"
          f"Central point [lat, lon]: {region_config.central_point}\n"
          f"Crosses meridian: {region_config.crosses_meridian}\n"
          f"Contour map mean precip range: {region_config.cm_mean_precip_range}\n"
          f"Time series mean precip range {region_config.ts_mean_precip_range}")

regions_info_dict = \
    {
    ##### CONTINENTAL UNITES STATES #####
    "CONUS": RegionPlottingConfiguration(
                region_extent = [-125, -66, 24, 51],
                figsize_sp = (15, 10), 
                figsize = (15, 7.75), # (15, 7.5)
                figsize_errors = (15, 7.75), # (15, 7.5)
                subplot_layout = "grid",
                cm_mean_precip_range = np.arange(0, 10.5, 0.5),
                ts_mean_precip_range = np.arange(0, 4.5, 0.5),
                central_point = [39.8, -98.6],
                crosses_meridian = False,
                ),
                
    ##### MAIN CONUS SUB-REGIONS #####
    "US-East": RegionPlottingConfiguration(
                region_extent = [-85, -66, 24, 51],
                figsize_sp = (10, 15), 
                figsize = (10, 12.5),
                figsize_errors = (10, 12.5),
                subplot_layout = "grid",
                cm_mean_precip_range = np.arange(0, 6.5, 0.5), 
                ts_mean_precip_range = np.arange(0, 7.0, 0.5),
                central_point = [37.3, -80.0],
                crosses_meridian = False,
                ),

    "US-Central": RegionPlottingConfiguration(
                region_extent = [-103, -85, 24, 51],
                figsize_sp = (11, 18), 
                figsize = (11, 14.5),
                figsize_errors = (11, 14.5),
                subplot_layout = "grid",
                cm_mean_precip_range = np.arange(0, 6.5, 0.5), 
                ts_mean_precip_range = np.arange(0, 6.5, 0.5),
                central_point = [38.5, -94.5],
                crosses_meridian = False,
                ),

    "US-Mountain": RegionPlottingConfiguration(
                #region_extent = [-117, -103, 28, 51], # Old extent for mountain states not explicitly included
                region_extent = [-120, -103, 28, 51],
                figsize_sp = (10, 14),
                figsize = (11, 13), # (11, 14)
                figsize_errors = (11, 13), # (11, 14)
                subplot_layout = "grid",
                cm_mean_precip_range = np.arange(0, 6.5, 0.5), 
                ts_mean_precip_range = np.arange(0, 3.0, 0.5),
                central_point = [39.1, -111.8],
                crosses_meridian = False,
                ),

    "US-WestCoast": RegionPlottingConfiguration( # Same domain as US-West, but verification will be confined to WA, OR, and CA
                region_extent = [-129, -113, 28, 51],
                figsize_sp = (10, 15), 
                figsize = (10, 12.5),
                figsize_errors = (10, 12.5),
                subplot_layout = "grid",
                cm_mean_precip_range = np.arange(0, 15, 1),
                ts_mean_precip_range = np.arange(0, 7.5, 0.5),
                central_point = [43.0, -121.0],
                crosses_meridian = False,
                ),

    ##### OTHER CONUS SUB-REGIONS #####
    "US-BigSioux": RegionPlottingConfiguration(
                region_extent = [-100, -87, 39, 49],
                figsize_sp = (13, 12), 
                figsize = (14, 11.5),
                figsize_errors = (12, 11.5),
                subplot_layout = "grid",
                cm_mean_precip_range = np.arange(0, 6.5, 0.5), 
                ts_mean_precip_range = np.arange(0, 3.5, 0.5), 
                central_point = [-999.0, -999.0],
                crosses_meridian = False,
                ),

    "US-Colorado": RegionPlottingConfiguration(
                region_extent = [-110, -100, 35, 43],
                figsize_sp = (13, 12), 
                figsize = (14, 12),
                figsize_errors = (14, 12),
                subplot_layout = "grid",
                cm_mean_precip_range = np.arange(0, 6.5, 0.5), 
                ts_mean_precip_range = np.arange(0, 3.5, 0.5), 
                central_point = [-999.0, -999.0],
                crosses_meridian = False,
                ),

    "US-ColoradoZoom": RegionPlottingConfiguration(
                region_extent = [-109.5, -101.5, 36.5, 41.5],
                figsize_sp = (14, 12), 
                figsize = (15, 13),
                figsize_errors = (15, 13),
                subplot_layout = "grid",
                cm_mean_precip_range = np.arange(0, 6.5, 0.5), 
                ts_mean_precip_range = np.arange(0, 3.5, 0.5), 
                central_point = [-999.0, -999.0],
                crosses_meridian = False,
                ),
    
    "US-GulfCoast": RegionPlottingConfiguration(
                region_extent = [-95, -80, 24, 32], 
                figsize_sp = (15, 11), 
                figsize = (15, 8),
                figsize_errors = (15, 8),
                subplot_layout = "grid",
                cm_mean_precip_range = np.arange(0, 6.5, 0.5), 
                ts_mean_precip_range = np.arange(0, 9.5, 0.5), 
                central_point = [-999.0, -999.0],
                crosses_meridian = False,
                ),

    "US-NorthEast": RegionPlottingConfiguration(
                region_extent = [-92, -67, 36, 49], 
                figsize_sp = (14, 10.5), 
                figsize = (15, 8.5),
                figsize_errors = (15, 8.5),
                subplot_layout = "grid",
                cm_mean_precip_range = np.arange(0, 6.5, 0.5), 
                ts_mean_precip_range = np.arange(0, 8.5, 0.5),
                central_point = [-999.0, -999.0],
                crosses_meridian = False,
                ),

    "US-SouthEast": RegionPlottingConfiguration(
                region_extent = [-95, -75, 24, 37], 
                figsize_sp = (15, 11), 
                figsize = (15, 9.5),
                figsize_errors = (15, 9.5),
                subplot_layout = "grid",
                cm_mean_precip_range = np.arange(0, 6.5, 0.5), 
                ts_mean_precip_range = np.arange(0, 8.5, 0.5),
                central_point = [-999.0, -999.0],
                crosses_meridian = False,
                ),
    
    "US-Tennessee": RegionPlottingConfiguration(
                region_extent = [-94, -80, 31, 40],
                figsize_sp = (13, 12), 
                figsize = (13, 12),
                figsize_errors = (13, 12),
                subplot_layout = "grid",
                cm_mean_precip_range = np.arange(0, 6.5, 0.5), 
                ts_mean_precip_range = np.arange(0, 3.5, 0.5), 
                central_point = [-999.0, -999.0],
                crosses_meridian = False,
                ),


    ##### OTHER PARTS OF THE WORLD #####
    # TODO: Additional region for equatorial+subtropical Africa only?
    "Africa": RegionPlottingConfiguration( 
                region_extent = [-20, 55, -38, 40],
                figsize_sp = (10, 12), 
                figsize = (11, 12), #(13, 15.5),
                figsize_errors = (11, 13), #(13, 15.5),
                subplot_layout = "grid",
                cm_mean_precip_range = np.arange(0, 17, 1),
                ts_mean_precip_range = np.arange(0, 4.5, 0.5),
                central_point = [1.0, 17.5],
                crosses_meridian = True,
                ),

    "Australia": RegionPlottingConfiguration(
                region_extent = [112, 156, -45, -9],
                figsize_sp = (13, 11), 
                figsize = (13, 10),
                figsize_errors = (13, 10),
                subplot_layout = "grid",
                cm_mean_precip_range = np.arange(0, 10.5, 0.5),
                ts_mean_precip_range = np.arange(0, 6.5, 0.5),
                central_point = [-999.0, -999.0],
                crosses_meridian = False,
                ),

    "EastPacAR": RegionPlottingConfiguration( # Extends into Pacific to capture ARs
                region_extent = [-150, -113, 28, 51],
                figsize_sp = (15, 10), 
                figsize = (15, 9),
                figsize_errors = (15, 9),
                subplot_layout = "grid",
                cm_mean_precip_range = np.arange(0, 10.5, 0.5), 
                ts_mean_precip_range = np.arange(0, 6.5, 0.5), 
                central_point = [-999.0, -999.0],
                crosses_meridian = False,
                ),

    "Europe": RegionPlottingConfiguration(
                region_extent = [-12, 50, 35, 72],
                figsize_sp = (15, 12), 
                figsize = (15, 10),
                figsize_errors = (15, 10),
                subplot_layout = "grid",
                cm_mean_precip_range = np.arange(0, 6.5, 0.5),
                ts_mean_precip_range = np.arange(0, 4.5, 0.5),
                central_point = [-999.0, -999.0],
                crosses_meridian = True,
                ),

    "MaritimeContinent": RegionPlottingConfiguration(
                region_extent = [90, 160, -18, 18],
                figsize_sp = (15, 11), 
                figsize = (15, 9),
                figsize_errors = (15, 9),
                subplot_layout = "grid",
                cm_mean_precip_range = np.arange(0, 21, 1),
                ts_mean_precip_range = np.arange(0, 13, 1),
                central_point = [-999.0, -999.0],
                crosses_meridian = False,
                ),

    "MJO": RegionPlottingConfiguration(
                region_extent = [70, 170, -18, 18],
                figsize_sp = (16, 10), 
                figsize = (16, 9),
                figsize_errors = (16, 9),
                subplot_layout = "grid",
                cm_mean_precip_range = np.arange(0, 21, 1),
                ts_mean_precip_range = np.arange(0, 13, 1),
                central_point = [-999.0, -999.0],
                crosses_meridian = False,
                ),

    "SouthAmerica": RegionPlottingConfiguration(
                region_extent = [-83, -33, -57, 13],
                figsize_sp = (9, 14), 
                figsize = (9, 12),
                figsize_errors = (9, 12),
                subplot_layout = "grid",
                cm_mean_precip_range = np.arange(0, 21, 1),
                ts_mean_precip_range = np.arange(0, 11, 1),
                central_point = [-999.0, -999.0],
                crosses_meridian = False,
                ),

    "WestPacJapan": RegionPlottingConfiguration(
                region_extent = [118, 178, 20, 50],
                figsize_sp = (14, 10), 
                figsize = (14, 8),
                figsize_errors = (14, 8),
                subplot_layout = "grid",
                cm_mean_precip_range = np.arange(0, 15.5, 0.5),
                ts_mean_precip_range = np.arange(0, 8, 1),
                central_point = [-999.0, -999.0],
                crosses_meridian = False,
                ),
    
    ##### GLOBAL, HEMISPHERES, AND TROPICS #####
    "Global": RegionPlottingConfiguration(
                region_extent = [-180, 180, -90, 90],
                figsize_sp = (15, 10), 
                figsize = (15, 9),
                figsize_errors = (15, 9),
                subplot_layout = "grid",
                cm_mean_precip_range = np.arange(0, 21, 1),
                ts_mean_precip_range = np.arange(0, 4.0, 0.5), 
                central_point = [0.0, 0.0],
                crosses_meridian = True,
                ),
    
    "NorthHem": RegionPlottingConfiguration(
                  region_extent = [-180, 180, 0, 90],
                  figsize_sp = (15, 6), 
                  figsize = (15, 9),
                  figsize_errors = (15, 9),
                  subplot_layout = "vertical",
                  cm_mean_precip_range = np.arange(0, 10.5, 0.5),
                  ts_mean_precip_range = np.arange(0, 4.0, 0.5), 
                  central_point = [45.0, 0.0],
                  crosses_meridian = True,
                  ),
    
    "NorthHemMidLat": RegionPlottingConfiguration(
                        region_extent = [-180, 180, 30, 65],
                        figsize_sp = (15, 3), 
                        figsize = (12, 5.3),
                        figsize_errors = (12, 6),
                        subplot_layout = "vertical",
                        cm_mean_precip_range = np.arange(0, 7.5, 0.5),
                        ts_mean_precip_range = np.arange(0, 4.0, 0.5), 
                        central_point = [47.5, 0],
                        crosses_meridian = True,
                        ),
    
    "SouthHem": RegionPlottingConfiguration(
                  region_extent = [-180, 180, -90, 0],
                  figsize_sp = (15, 6), 
                  figsize = (15, 9),
                  figsize_errors = (15, 9),
                  subplot_layout = "vertical",
                  cm_mean_precip_range = np.arange(0, 8.5, 0.5),
                  ts_mean_precip_range = np.arange(0, 4.0, 0.5), 
                  central_point = [-45,0, 0.0],
                  crosses_meridian = True,
                  ),
    
    "SouthHemMidLat": RegionPlottingConfiguration(
                        region_extent = [-180, 180, -65, -30],
                        figsize_sp = (15, 3), 
                        figsize = (12, 5.3),
                        figsize_errors = (12, 6),
                        subplot_layout = "vertical",
                        cm_mean_precip_range = np.arange(0, 7.5, 0.5),
                        ts_mean_precip_range = np.arange(0, 4.0, 0.5), 
                        central_point = [-47.5, 0.0],
                        crosses_meridian = True,
                        ),
    
    "Tropics": RegionPlottingConfiguration(
                 region_extent = [-180, 180, -30, 30],
                 figsize_sp = (17, 6), 
                 figsize = (17, 7.5),
                 figsize_errors = (17, 8.5),
                 subplot_layout = "vertical",
                 cm_mean_precip_range = np.arange(0, 15.5, 0.5),
                 ts_mean_precip_range = np.arange(2.0, 5.5, 0.5), 
                 central_point = [0.0, 0.0],
                 crosses_meridian = True,
                 ),
    }

# Return dot-seperated string of data names from the keys of a dictionary
# of the form {data_name1: da1, data_name2: da2,...., data_nameN: daN}
def get_data_names_str(data_dict):
    return "".join(f"{key}." for key in data_dict.keys())

# Map three-letter month strings to the correponding numerical month of the year.
def month_string_to_month_number(month_string):
    month_to_number_dict = {
                            "JAN" :  1,
                            "FEB" :  2,
                            "MAR" :  3,
                            "APR" :  4,
                            "MAY" :  5,
                            "JUN" :  6,
                            "JUL" :  7,
                            "AUG" :  8,
                            "SEP" :  9,
                            "OCT" : 10,
                            "NOV" : 11,
                            "DEC" : 12,
                           }

    return month_to_number_dict[month_string]

# Map three-letter season strings to the corresponding numerical season (chose
# to start counting with DJF = 1).
def season_string_to_season_number(season_string):
    season_to_number_dict = {
                             "DJF" :  1,
                             "MAM" :  2,
                             "JJA" :  3,
                             "SON" :  4,
                            }

    return season_to_number_dict[season_string]

# Map a datetime object to the common month string
def dtime_to_month_string(dtime):
    month = dtime.month
    month_to_month_string_dict = {
                                  1  : "JAN",
                                  2  : "FEB",
                                  3  : "MAR",
                                  4  : "APR",
                                  5  : "MAY",
                                  6  : "JUN",
                                  7  : "JUL",
                                  8  : "AUG",
                                  9  : "SEP",
                                  10 : "OCT",
                                  11 : "NOV",
                                  12 : "DEC",
                                 }

    return month_to_month_string_dict[month]

# Map a datetime object to the common season string
def dtime_to_season_string(dtime):
    month = dtime.month
    month_to_season_string_dict = {
                                   1  : "DJF",
                                   2  : "DJF",
                                   3  : "MAM",
                                   4  : "MAM",
                                   5  : "MAM",
                                   6  : "JJA",
                                   7  : "JJA",
                                   8  : "JJA",
                                   9  : "SON",
                                   10 : "SON",
                                   11 : "SON",
                                   12 : "DJF",
                                  }

    return month_to_season_string_dict[month]

# Returns a 12-element list of three-letter months strings.
def construct_monthly_string_list():
    return ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

# Returns a 4-element list of three-letter season strings.
def construct_seasonal_string_list():
    return ["DJF", "MAM", "JJA", "SON"]

# Dictionary mapping dataset names to colors to use for the respective
# lines on timeseries plots
datasets_colors_dict = {"AORC":                                     "blue",
                        "CONUS404":                               "purple",
                        "ERA5":                                      "red",
                        "GFS":                                      "gold",
                        "HRRR":                                     "cyan",
                        "HRRR-06hr":                           "slategrey",
                        "IMERG":                                   "green",
                        "NestedEagle":                             "black",
                        "NestedEagle-fhr06":                       "black",
                        "NestedEagle-fhr12":                       "black",
                        "NestedEagle-fhr24":                       "black",
                        "NestedEagle-fhr48":                       "black",
                        "NestedReplay":                             "teal",
                        "NestedReplayBetaCu":                  "slategrey",
                        "NestedReplayDxMfluxImidgf0ForceOff":      "brown",
                        "NestedReplayDxMflux":                     "coral",
                        "NestedReplaySAS":                     "turquoise",
                        "NestedReplayNoConv":             "mediumseagreen",
                        "NRGl":                                     "teal",
                        "NRGlBetaCu":                          "slategrey",
                        "NRGlDxMfluxImidgf0ForceOff":              "brown",
                        "NRGlDxMflux":                             "coral",
                        "NRGlSAS":                             "turquoise",
                        "NRGlNoConv":                     "mediumseagreen",
                        "NestedReplayPredictor":               "cadetblue",
                        "Replay":                                 "orange",
                       }

# Potential replacement if want different linestyles for different versions 
# of the same dataset (e.g, NestedReplaySAS vs. NestedReplayNoConv)
@dataclasses.dataclass
class DatasetPlotStyle:
    color: str
    linestyle: str 
datasets_linestyles_dict = {"AORC":                  DatasetPlotStyle(color = "blue", linestyle = "solid"), 
                            "CONUS404":              DatasetPlotStyle(color = "purple", linestyle = "solid"),
                            "ERA5":                  DatasetPlotStyle(color = "red", linestyle = "solid"),
                            "GFS":                   DatasetPlotStyle(color = "gold", linestyle = "solid"),
                            "HRRR":                  DatasetPlotStyle(color = "cyan", linestyle = "solid"),
                            "HRRR-06hr":             DatasetPlotStyle(color = "slategrey", linestyle = "solid"),
                            "IMERG":                 DatasetPlotStyle(color = "green", linestyle = "solid"),
                            "NestedEagle":           DatasetPlotStyle(color = "black", linestyle = "solid"),
                            "NestedEagle-fhr06":     DatasetPlotStyle(color = "black", linestyle = "solid"),
                            "NestedEagle-fhr12":     DatasetPlotStyle(color = "black", linestyle = "solid"),
                            "NestedEagle-fhr24":     DatasetPlotStyle(color = "black", linestyle = "solid"),
                            "NestedEagle-fhr48":     DatasetPlotStyle(color = "black", linestyle = "solid"),
                            "NestedReplay":          DatasetPlotStyle(color = "teal", linestyle = "solid"),
                            "NestedReplaySAS":       DatasetPlotStyle(color = "teal", linestyle = "dashed"),
                            "NestedReplayNoConv":    DatasetPlotStyle(color = "teal", linestyle = "dotted"),
                            "NestedReplayPredictor": DatasetPlotStyle(color = "cadetblue", linestyle = "solid"),
                            "Replay":                DatasetPlotStyle(color = "orange", linestyle = "solid"),
                           }

# Get the name of the time dimension from a data array
# looking through all time dimensions defined in utilities.py
def get_time_dimension_name(data_array):
    if (utils.time_dim_str in data_array.dims):
        return utils.time_dim_str
    elif (utils.period_begin_time_dim_str in data_array.dims):
        return utils.period_begin_time_dim_str
    elif (utils.period_end_time_dim_str in data_array.dims):
        return utils.period_end_time_dim_str
    elif (utils.months_dim_str in data_array.dims):
        return utils.months_dim_str
    elif (utils.seasons_dim_str in data_array.dims):
        return utils.seasons_dim_str
    elif (utils.annual_dim_str in data_array.dims):
        return utils.annual_dim_str
    elif (utils.full_period_dim_str in data_array.dims):
        return utils.full_period_dim_str
    elif (utils.common_month_dim_str in data_array.dims):
        return utils.common_month_dim_str
    elif (utils.common_season_dim_str in data_array.dims):
        return utils.common_season_dim_str
    else:
        print("Error: Can't find a valid time dimension within dimensions {}")
        sys.exit(1)

# Create the CONUS mask, excluding AK and HI, to only include data over the lower 48 states (i.e., the CONUS)
# Each state+DC is represented by an integer between 0 and 50 (clunky, but that's the way it is)
# So there are 51 "states" total (regionmask refers to each stats as a region)
def create_conus_mask(data_array):
    # Define a regionmask object representing all 50 US states
    states = regionmask.defined_regions.natural_earth_v5_0_0.us_states_50

    # Create a mask valid within all 50 US states, in the form of an xarray DataArray
    # with coordinates the same as <data_array>. The mask method determines which grid
    # points of the array lie within the region. Grid points outside of the region are set to NaN.
    # Gridpoints within the region are encoded with an integer representing the region.
    # In this case, each of the 50 states is defined with a different integer. 
    mask = states.mask(data_array)

    # Get the integers representing the non-CONUS states Alaska and Hawaii (don't want to include data over these states in CONUS stats) 
    AK_index = states.map_keys("Alaska")
    HI_index = states.map_keys("Hawaii")

    # Convert the mask to a boolean DataArray that is True over all the states except AK and HI
    # We'll use this boolean DataArray to pull out only data from the precip DataArrays within the lower-48 (CONUS) states 
    mask_conus = (mask != AK_index) & (mask != HI_index) & (mask >= 0) 

    return mask_conus

# Follow the logic above for the CONUS mask, but for only the states within the US-Mountain region. 
def create_mountain_states_mask(data_array):
    states = regionmask.defined_regions.natural_earth_v5_0_0.us_states_50
    mask = states.mask(data_array)
    ID_index = states.map_keys("Idaho")
    NV_index = states.map_keys("Nevada")
    UT_index = states.map_keys("Utah")
    AZ_index = states.map_keys("Arizona")
    NM_index = states.map_keys("New Mexico")
    CO_index = states.map_keys("Colorado")
    WY_index = states.map_keys("Wyoming")
    MT_index = states.map_keys("Montana")
    ND_index = states.map_keys("North Dakota")
    SD_index = states.map_keys("South Dakota")
    NE_index = states.map_keys("Nebraska")
    TX_index = states.map_keys("Texas")
    mask_west_coast_states = (mask == ID_index) | (mask == NV_index) | (mask == UT_index) | \
                             (mask == AZ_index) | (mask == NM_index) | (mask == CO_index) | \
                             (mask == WY_index) | (mask == MT_index) | (mask == ND_index) | \
                             (mask == SD_index) | (mask == NE_index) | (mask == TX_index)   

    return mask_west_coast_states

# Follow the logic above for the CONUS mask, but for only the states of WA, OR, and CA
def create_west_coast_states_mask(data_array):
    states = regionmask.defined_regions.natural_earth_v5_0_0.us_states_50
    mask = states.mask(data_array)
    WA_index = states.map_keys("Washington")
    OR_index = states.map_keys("Oregon")
    CA_index = states.map_keys("California")
    mask_west_coast_states = (mask == WA_index) | (mask == OR_index) | (mask == CA_index)

    return mask_west_coast_states

# Follow the logic above for the CONUS mask, but for all countries that are part of additional
# continental regions. NOTE: France is included in the South America list to capture French Guiana. 
def create_continent_mask(data_array, region):
    countries = regionmask.defined_regions.natural_earth_v5_1_2.countries_50
    mask = countries.mask(data_array)

    country_list_file = os.path.join(utils.home_dir, "regionmask_country_lists", f"{region}.csv")
    country_list = [i.strip() for i in open(country_list_file).readlines()]
    country_indices = [countries.map_keys(i) for i in country_list]

    for i, country_index in enumerate(country_indices):
        if (i > 0):
            mask_continent = (mask == country_index) | (mask_continent == True) 
        else:
            mask_continent = (mask == country_index)

    return mask_continent

def set_cbar_labels_rotation_and_fontsize(region_plot_config, plot_levels, num_da, for_single_cbar = False):
    rotation = 0
    fontsize = 14

    # If for a smaller colorbar on a subplot, set small rotation and reduce the fontsize
    if not(for_single_cbar):
        rotation = 15
        fontsize = 13
        
    # If the number of levels is large, set a larger rotation and reduce the fontsize
    if (len(plot_levels) >= 10):
        rotation = 30
        fontsize = 12 

        # For subplot colorbars (for_single_cbar = False), if the number of plot levels 
        # is large AND the figure is taller than it is wide, reduce the fontsize and
        # increase the rotation further. Otherwise, color bar labels will be too squished
        # on "tall" plots (e.g., US-WestCoast).
        if not(for_single_cbar):
            if (num_da > 1):
                figsize_ratio = region_plot_config.figsize[0]/region_plot_config.figsize[1]
            else:
                figsize_ratio = region_plot_config.figsize_sp[0]/region_plot_config.figsize_sp[1]
            if (figsize_ratio < 1.0):
                rotation = 60 # was 75
                fontsize = 11 # was 10

    return rotation, fontsize

# Convert whole-number plot levels into integers for neater tick labelling
def set_plot_levels_whole_numbers_to_integers(plot_levels):
    tick_labels = []
    for i in plot_levels:
        if (i % 1 == 0):
            tick_labels.append(int(i))
        else:
            tick_labels.append(i)

    return tick_labels

def how_to_create_precip_plot_levels():
    print("NOTE: Returns colormap, plot_levels, color_list")
    print('create_precip_plot_levels(temporal_res = 1,\n'
          '                          version = 1,\n'
          '                          use_existing_cmap = False,\n'
          '                          existing_cmap = "viridis"\n'
          '                          add_near_zero_first_level = False,\n'
          '                          near_zero_first_level = 0.001)') 

def create_precip_plot_levels(temporal_res = 1,
                              version = 1,
                              use_existing_cmap = False,
                              existing_cmap = "viridis",
                              add_near_zero_first_level = False,
                              near_zero_first_level = 0.001):
    # The Version 2 color list tends to work better for longer
    # accumulations (24 hours+), but it still works well for hourly.
    # Version 2 is also similar to typical reflectivity (dBZ) color scales.
    if (version == 1): # Version 1 
        color_list = ["white",
                      "lawngreen", "limegreen", "green", "darkgreen", # Greens
                      "teal", "royalblue", "dodgerblue", "deepskyblue", # Blues
                      "mediumorchid", "darkorchid", "darkviolet", "purple", # Purples
                      "darkred", "firebrick", "crimson", "red", # Reds
                      "orangered", "darkorange", "gold", # Oranges
                      ]
    else: # Version 2
        color_list = ["white",
                     "cyan", "deepskyblue", "dodgerblue", "blue", # Blues
                     "lawngreen", "limegreen", "green", "darkgreen", # Greens
                     "yellow", "gold", "goldenrod", "darkgoldenrod", # Yellows
                     "lightcoral", "crimson", "firebrick", "maroon", # Reds
                     "plum", "violet", "purple" # Purples
                     ]

    base_levels = np.array([
                           0,
                           0.25, 0.5, 0.75, 1,
                           2, 3, 4, 5,
                           6, 8, 10, 12,
                           14, 16, 20, 25,
                           30, 35, 40
                           ])

    if add_near_zero_first_level:
        color_list = [ color_list[0] ] + color_list
        color_list[1] = "paleturquoise"

        base_levels = np.concatenate(( np.array([0]), base_levels  ))
        base_levels[1] = near_zero_first_level

    match temporal_res:
        case 1:
            plot_levels = base_levels
        case 3:
            plot_levels = base_levels
        case 6:
            plot_levels = base_levels
        case 12:
            plot_levels = base_levels * 2 
        case 24:
            plot_levels = base_levels * 2 # Was 2.5
        case 48:
            plot_levels = base_levels * 3 # Was 4
        case 72:
            plot_levels = base_levels * 4 # Was 6
        case 96:
            plot_levels = base_levels * 5 # Was 8
        case 120:
            plot_levels = base_levels * 6 # Was 10
        case 144:
            plot_levels = base_levels * 8 # Was 12
        case _: 
            plot_levels = base_levels * 10 # Was 15

    # Cut off the lowest value from plot levels, so values below it show up as white
    if use_existing_cmap:
        return existing_cmap, plot_levels[1:], existing_cmap
    else: 
        colormap = mpl.colors.ListedColormap(color_list)
        norm = mpl.colors.BoundaryNorm(plot_levels, colormap.N)

    return colormap, plot_levels, color_list 

# Old precip plot levels and color lists
def OLD_create_precip_plot_levels(temporal_res = 1, bound_to_key_thresholds = True):
    if bound_to_key_thresholds:
        base_levels = np.array([
                                0, # white
                                0.25, 1, 2, 3, 4, # greens
                                5, 6, 7, 8, # blues
                                9, 10, 13, 16, # purples --> reds
                                19, 22, 25]) # yellows
        # For 24-hour accumulated precip
        base_levels24 = np.array([0, # white
                                  1, 2, 4, 6, 8, # greens
                                  10, 12, 15, 20, # blues
                                  25, 30, 35, 40, # purples --> reds
                                  50, 60, 75]) # yellows

        color_list = ["white",
                      "palegreen", "limegreen", "green", "darkgreen", "teal",
                      "royalblue", "dodgerblue", "deepskyblue", "cyan", 
                      "mediumorchid", "purple", "crimson", "red",
                      "orange", "gold", "yellow"
                     ]
                      #"darkred", "crimson", "orangered", "darkorange", "gold"]
    else:
        base_levels = np.array([0, 
                                0.2, 0.4, 0.6, 0.8, 1, 2,
                                3, 4, 5, 6, 7, 8, 9,
                                10, 12, 14, 16,
                                18, 20, 24, 28, 32,
                                36, 40])
        # For 24-hour accumulated precip
        base_levels24 = np.array([0, 
                                  0.25, 0.5, 0.75, 1, 2, 3,
                                  4, 5, 6, 7, 8, 9, 10, 
                                  12, 15, 20, 25, 
                                  30, 35, 40, 45, 60,
                                  75, 100])
        color_list = [
                      "white",
                      "palegreen", "lawngreen", "lime", "limegreen", "green", "darkgreen", # Greens
                      "teal", "royalblue", "dodgerblue", "deepskyblue", "darkturquoise", "turquoise", "cyan", # Blues
                      "mediumorchid", "darkorchid", "darkviolet", "purple", # Purples
                      "darkred", "firebrick", "crimson", "red", "orangered", # Reds
                      "darkorange", "gold", # Oranges
                      ]

    match temporal_res:
        case 1:
            plot_levels = base_levels
        case 3:
            plot_levels = base_levels
        case 6:
            plot_levels = base_levels * 2 
        case 12:
            plot_levels = base_levels * 2 
        case 24:
            plot_levels = base_levels24 
            #plot_levels = base_levels * 3
        case 48:
            plot_levels = base_levels * 4
        case 72:
            plot_levels = base_levels * 4
        case 96:
            plot_levels = base_levels * 8
        case 120:
            plot_levels = base_levels * 10
        case 144:
            plot_levels = base_levels * 10
        case 168:
            plot_levels = base_levels * 17.5
    
    colormap = mpl.colors.ListedColormap(color_list)
    norm = mpl.colors.BoundaryNorm(plot_levels, colormap.N)

    return colormap, plot_levels, norm

##### Very old precip color lists and plot levels 
# Hourly
#bounds = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 16, 19, 22, 25, 30, 35, 40, 45, 50]
#color_list = ["white", "lightgreen", "lawngreen", "lime", "limegreen", "green",
#              "teal", "turquoise", "aqua", "lightskyblue",
#              "deepskyblue", "blue", "darkblue", "slateblue", "purple",
#              "magenta", "violet", "plum", "pink", "lightcoral",
#              "crimson", "red", "orange", "gold", "yellow"
#             ]

# Hourly, bound to key thresholds
#bounds = [0, 0.25, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 16, 19, 22, 25]
#color_list = ["white",
#              "lightgreen", "limegreen", "green", "darkgreen", "teal",
#              "aqua", "deepskyblue", "blue", "darkblue",
#              "slateblue", "purple", "crimson", "red",
#              "orange", "gold", "yellow"]

# 24-hourly
#bounds = [0.0, 0.5, 1.0, 1.5, 2, 3, 6, 9, 12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 85, 95, 105]
#color_list = ["white", "lightgreen", "lawngreen", "lime", "limegreen", "green",
#              "teal", "turquoise", "aqua", "lightskyblue",
#              "deepskyblue", "blue", "darkblue", "slateblue", "purple",
#              "magenta", "violet", "plum", "pink", "lightcoral",
#              "crimson", "red", "orange", "gold", "yellow"
#             ]

# 24-hourly, bound to key thresholds
# bounds = [0, 2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 75]
#bounds = [0, 1, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 50, 60, 75]
#color_list = ["white",
#              "lightgreen", "limegreen", "green", "darkgreen", "teal",
#              "aqua", "deepskyblue", "blue", "darkblue",
#              "slateblue", "purple", "crimson", "red",
#              "orange", "gold", "yellow"]

# Potential alternative colormap (untested)
#color_list = ["white",
#              "lightyellow", "oldlace", "tan", "peachpuff",
#              "lightgreen", "green", "forestgreen",
#              "deepskyblue", "dodgerblue", "blue",
#              "orange", "red", "violet", "magenta", "purple"]

# Plot limits to use for contour maps for different variables
# and for each accumulation period for precipitation.
def variable_plot_limits(var_name, temporal_res = "native"):
    match var_name:
        case "tmp2m":
            return np.arange(-15, 45, 5) # Celsius 
        case _: # For any other variable, assume it's precipitation 
            match temporal_res:
                case 1:
                    return np.arange(0, 32, 2) # mm 
                case 3:
                    return np.arange(0, 55, 5)
                case 6:
                    return np.arange(0, 65, 5)
                case 12:
                    return np.arange(0, 85, 5) 
                case 24:
                    return np.arange(0, 85, 5)
                case 48:
                    return np.arange(0, 125, 5)
                case 72:
                    return np.arange(0, 210, 10)
                case 120:
                    return np.arange(0, 310, 10)
                case 168:
                    return np.arange(0, 420, 20)
                case _: # For example, "native" resolution
                    return np.arange(0, 32, 2)

# Plot limits to use for contour maps of percentiles for different variables
# and for each accumulation period for precipitation.
# TODO: Use different plot limits for 95th and 99th percentiles (and additional percentiles)?
def variable_pctl_plot_limits(var_name, temporal_res = "native"):
    match var_name:
        case utils.accum_precip_var_name:
            match temporal_res:
                case "native":
                    return np.arange(0, 11, 1) # mm 
                case 1:
                    return np.arange(0, 11, 1) 
                case 3:
                    return np.arange(0, 22, 2)
                case 6:
                    return np.arange(0, 36, 5)
                case 12:
                    return np.arange(0, 55, 5) 
                case 24:
                    return np.arange(0, 110, 10)
                case _:
                    return np.arange(0, 11, 1)
        case "tmp2m":
            return np.arange(280, 330, 5) # Kelvin
        case _:
            print(f"Variable percentile plot limits not defined for var {var_name}")
            sys.exit(1)

# Set bins for error box plots, histograpms, PDFs etc. based on region
# and the longer-term time period being covered (2002-2021 without HRRR;
# 2015-2021 with HRRR)
# These bins are based on range of AORC data
def set_hist_bins_by_region(region, hrrr_time_period = False):
    match region:
        case "CONUS":
            if hrrr_time_period:
                return np.array([0, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 640])
            else: 
                return np.array([0, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 640])
        case "US-East":
            if hrrr_time_period:
                return np.array([0, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300])
            else: 
                return np.array([0, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300])
        case "US-Central":
            if hrrr_time_period:
                return np.array([0, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 640])
            else: 
                return np.array([0, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 640])
        case "US-Mountain":
            if hrrr_time_period:
                return np.array([0, 10, 20, 30, 40, 50, 75, 100, 150])
            else: 
                return np.array([0, 10, 20, 30, 40, 50, 75, 100, 150])
        case "US-WestCoast":
            if hrrr_time_period:
                return np.array([0, 10, 20, 30, 40, 50, 75, 100, 150, 200])
            else: 
                return np.array([0, 10, 20, 30, 40, 50, 75, 100, 150, 200])

@dataclasses.dataclass
class CDF:
    quantiles: float
    values: float

def create_cdf(data_array):
    quantiles = np.concatenate(( np.arange(0.0, 1.0, 0.01), np.arange(0.991, 1.0, 0.001) ))
    values = np.array([data_array.quantile(i).item() for i in quantiles])

    return CDF(quantiles = quantiles, values = values)

def how_to_plot_precip_cdf():
    print("plot_precip_cdf(data_dict, plot_name, fig_name,\n"
          "                valid_dtime = None,\n"
          "                xticks = np.arange(0, 85, 5),\n"
          "                yticks = np.arange(0, 1.1, 0.1),\n"
          "                xlims = [0, 80],\n"
          "                ylims = [0, 1.0],\n"
          "                xticks_rotation = 0,\n"
          "                skip_nearest = False)")

def plot_precip_cdf(data_dict, plot_name, fig_name,
                    valid_dtime = None,
                    xticks = np.arange(0, 85, 5),
                    yticks = np.arange(0, 1.1, 0.1),
                    xlims = [0, 80],
                    ylims = [0, 1.0],
                    xticks_rotation = 0,
                    skip_nearest = False):
    cdf_dict = {} 
    for data_name, da in data_dict.items():
        cdf_dict[data_name] = create_cdf(da) 

    plt.figure(figsize = (10, 10))
    plt.grid(True, linewidth = 0.5)
    plt.xlabel("Precip amount (mm)", size = 15)
    plt.ylabel("Probability", size = 15)
    plt.xticks(xticks, xticks, fontsize = 15, rotation = xticks_rotation)
    plt.yticks(yticks, [f"{ytick:0.2f}" for ytick in yticks], fontsize = 15)
    plt.xlim(xlims)
    plt.ylim(ylims)
    for cdf_name, cdf in cdf_dict.items():
        # For CDFs of different interpolation methods, can be useful to skip plotting of nearest-neighbor output
        # (often qualitatively similar to bilinear interoplation output).
        if (skip_nearest and cdf_name == "Nearest"):
            continue 
        plt.plot(cdf_dict[cdf_name].values, cdf_dict[cdf_name].quantiles, linewidth = 2, label = cdf_name)
    plt.legend(loc = "best", prop = {"size": 15})

    title_string = f"CDFs {plot_name}"
    fig_name = f"CDF.{fig_name}"
    if valid_dtime is not None:
        title_string += f", valid {valid_dtime:%Y%m%d}"
        fig_name += f".{valid_dtime:%Y%m%d}"
    plt.title(title_string, size = 15)
    fig_name += ".png"
    fig_fpath = os.path.join(utils.plot_output_dir, fig_name)
    plt.tight_layout() 
    print(f"Saving {fig_fpath}") 
    plt.savefig(fig_fpath)

# Label only every-other tick of a colorbar and use "" (blank) for the other ticks
# This logic is necessary since the lists of colorbar ticks and tick labels must have the same length
# Useful for small colorbars on multi-panel subplots
def create_sparse_cbar_ticks(plot_levels, make_integers = True):
    if make_integers:
        plot_levels = set_plot_levels_whole_numbers_to_integers(plot_levels)

    cbar_tick_labels = []
    for level in plot_levels[::2]:
        cbar_tick_labels.append(level)
        cbar_tick_labels.append("")

    if (len(cbar_tick_labels) == len(plot_levels) + 1):
        cbar_tick_labels = cbar_tick_labels[:-1]

    return cbar_tick_labels

def create_gridded_subplots(num_da, proj = ccrs.PlateCarree(), layout = "grid"):
        if (num_da == 1):
            return [plt.axes(projection = proj)]

        axes_list = []
        if (layout == "vertical"):
            for i in range(1, num_da + 1):
                axes_list.append(plt.subplot(num_da, 1, i, projection = proj))
            return axes_list 
        elif (layout == "horizontal"):
            for i in range(1, num_da + 1):
                axes_list.append(plt.subplot(1, num_da, i, projection = proj))
            return axes_list 
        else: # grid
            for i in range(1, num_da + 1):
                if (num_da == 2):
                    axes_list.append(plt.subplot(int(num_da/2), 2, i, projection = proj)) 
                elif (num_da % 2) == 0:
                    axes_list.append(plt.subplot(2, int(num_da/2), i, projection = proj))
                else:
                    if (i == 1):
                        idx1 = i
                        idx2 = i + 1
                    axes_list.append(plt.subplot(2, num_da + 1, (idx1, idx2), projection = proj))
                    if (idx1 != num_da):
                        idx1 += 2
                        idx2 += 2 
                    else:
                        idx1 += 3 
                        idx2 += 3 

            return axes_list 

# Create subplots using subplot2grid for use in multi-panel contour map.
def OLD_create_gridded_subplots(num_da, proj, single_colorbar = True):
    cbar_ax = None
    match num_da:
        case 1:
            axes_list = [plt.axes(projection = proj)]
        case 2:
            if single_colorbar:
                axes_list = [
                            plt.subplot2grid((5, 4), (0, 0), colspan = 2, rowspan = 4, projection = proj),
                            plt.subplot2grid((5, 4), (0, 2), colspan = 2, rowspan = 4, projection = proj),
                            ]
                cbar_ax = plt.subplot2grid((5, 4), (4, 1), colspan = 2, rowspan = 1)
            else:
                axes_list = [
                            plt.subplot2grid((4, 4), (0, 0), colspan = 2, rowspan = 4, projection = proj),
                            plt.subplot2grid((4, 4), (0, 2), colspan = 2, rowspan = 4, projection = proj),
                            ]
        case 3:
            if single_colorbar:
                axes_list = [
                            plt.subplot2grid((11, 4), (0, 0), colspan = 2, rowspan = 5, projection = proj),
                            plt.subplot2grid((11, 4), (0, 2), colspan = 2, rowspan = 5, projection = proj),
                            plt.subplot2grid((11, 4), (5, 1), colspan = 2, rowspan = 5, projection = proj),
                            ]
                cbar_ax = plt.subplot2grid((11, 4), (10, 1), colspan = 2, rowspan = 1)
            else:
                axes_list = [
                            plt.subplot2grid((8, 4), (0, 0), colspan = 2, rowspan = 4, projection = proj),
                            plt.subplot2grid((8, 4), (0, 2), colspan = 2, rowspan = 4, projection = proj),
                            plt.subplot2grid((8, 4), (4, 1), colspan = 2, rowspan = 4, projection = proj),
                            ]
        case 4:
            if single_colorbar:
                axes_list = [
                            plt.subplot2grid((11, 4), (0, 0), colspan = 2, rowspan = 5, projection = proj),
                            plt.subplot2grid((11, 4), (0, 2), colspan = 2, rowspan = 5, projection = proj),
                            plt.subplot2grid((11, 4), (5, 0), colspan = 2, rowspan = 5, projection = proj),
                            plt.subplot2grid((11, 4), (5, 2), colspan = 2, rowspan = 5, projection = proj),
                            ]
                cbar_ax = plt.subplot2grid((11, 4), (10, 1), colspan = 2, rowspan = 1)
            else:
                axes_list = [
                            plt.subplot2grid((8, 4), (0, 0), colspan = 2, rowspan = 4, projection = proj),
                            plt.subplot2grid((8, 4), (0, 2), colspan = 2, rowspan = 4, projection = proj),
                            plt.subplot2grid((8, 4), (4, 0), colspan = 2, rowspan = 4, projection = proj),
                            plt.subplot2grid((8, 4), (4, 2), colspan = 2, rowspan = 4, projection = proj),
                            ]
        case 5:
            if single_colorbar:
                axes_list = [
                            plt.subplot2grid((11, 6), (0, 0), colspan = 2, rowspan = 5, projection = proj),
                            plt.subplot2grid((11, 6), (0, 2), colspan = 2, rowspan = 5, projection = proj),
                            plt.subplot2grid((11, 6), (0, 4), colspan = 2, rowspan = 5, projection = proj),
                            plt.subplot2grid((11, 6), (5, 1), colspan = 2, rowspan = 5, projection = proj),
                            plt.subplot2grid((11, 6), (5, 3), colspan = 2, rowspan = 5, projection = proj),
                            ]
                cbar_ax = plt.subplot2grid((11, 6), (10, 1), colspan = 4, rowspan = 1)
            else:
                axes_list = [
                            plt.subplot2grid((8, 6), (0, 0), colspan = 2, rowspan = 4, projection = proj),
                            plt.subplot2grid((8, 6), (0, 2), colspan = 2, rowspan = 4, projection = proj),
                            plt.subplot2grid((8, 6), (0, 4), colspan = 2, rowspan = 4, projection = proj),
                            plt.subplot2grid((8, 6), (4, 1), colspan = 2, rowspan = 4, projection = proj),
                            plt.subplot2grid((8, 6), (4, 3), colspan = 2, rowspan = 4, projection = proj),
                            ]
        case 6:
            if single_colorbar:
                axes_list = [
                            plt.subplot2grid((11, 6), (0, 0), colspan = 2, rowspan = 5, projection = proj),
                            plt.subplot2grid((11, 6), (0, 2), colspan = 2, rowspan = 5, projection = proj),
                            plt.subplot2grid((11, 6), (0, 4), colspan = 2, rowspan = 5, projection = proj),
                            plt.subplot2grid((11, 6), (5, 0), colspan = 2, rowspan = 5, projection = proj),
                            plt.subplot2grid((11, 6), (5, 2), colspan = 2, rowspan = 5, projection = proj),
                            plt.subplot2grid((11, 6), (5, 4), colspan = 2, rowspan = 5, projection = proj),
                            ]
                cbar_ax = plt.subplot2grid((11, 6), (10, 1), colspan = 4, rowspan = 1)
            else:
                axes_list = [
                            plt.subplot2grid((8, 6), (0, 0), colspan = 2, rowspan = 4, projection = proj),
                            plt.subplot2grid((8, 6), (0, 2), colspan = 2, rowspan = 4, projection = proj),
                            plt.subplot2grid((8, 6), (0, 4), colspan = 2, rowspan = 4, projection = proj),
                            plt.subplot2grid((8, 6), (4, 0), colspan = 2, rowspan = 4, projection = proj),
                            plt.subplot2grid((8, 6), (4, 2), colspan = 2, rowspan = 4, projection = proj),
                            plt.subplot2grid((8, 6), (4, 4), colspan = 2, rowspan = 4, projection = proj),
                            ]
        case _:
            print("Error: create_gridded_subplots not currently configured to handle {num_da}-paneled subplot")
            sys.exit(1)
   
    return axes_list, cbar_ax

# FIXME: Doesn't position colorbar axis correctly; check numbers and math
def create_cbar_axis_based_on_subplot_axes(fig, num_subplots, axes_list):
    if (num_subplots == 1) or (num_subplots == 3):
        axis = axes_list[-1]
        axis_position = axis.get_position()
        
        xposition = axis_position.x0 - 0.04
        yposition = axis_position.y0 - 0.08
        cbar_axis_width = (axis_position.x1 - axis_position.x0) + 0.08
    elif (num_subplots == 2) or (num_subplots == 4) or (num_subplots == 5):
        axis_left = axes_list[num_subplots - 2]
        axis_right = axes_list[num_subplots - 1]
        axis_left_position = axis_left.get_position()
        axis_right_position = axis_right.get_position()
        axis_left_midpoint = 0.5 * (axis_left_position.x1 + axis_left_position.x0)
        axis_right_midpoint = 0.5 * (axis_right_position.x1 + axis_right_position.x0)

        xposition = axis_left_midpoint
        yposition = axis_left_position.y0 - 0.08
        cbar_axis_width = axis_right_midpoint - axis_left_midpoint 
    elif (num_subplots == 6):
        axis = axes_list[-2]
        axis_position = axis.get_position()

        xposition = axis_position.x0 - 0.04
        yposition = axis_position.y0 - 0.08
        cbar_axis_width = (axis_position.x1 - axis_position.x0) + 0.08 
    else:
        print(f"Error: Colorbar axes for {num_subplots}-paneled subplot not yet supported")
        sys.exit(1)

    return fig.add_axes([xposition, # Lower left horizontal position
                         yposition, # Lower left vertical position
                         cbar_axis_width, # Width in x (horizontal) direction 
                         0.04]) # Height in y (vertical) direction

def set_figsize_based_on_num_da(num_da, region_plotting_config):
    if (num_da == 1):
        return region_plotting_config.figsize_sp
    else:
        return region_plotting_config.figsize

def add_cartopy_features_to_map_proj(axis, region, data_proj, draw_labels = False):
        axis.coastlines()
        axis.set_extent(regions_info_dict[region].region_extent, crs = data_proj)
        axis.add_feature(cfeature.BORDERS)
        if ("US" in region):
            axis.add_feature(cfeature.STATES)
        gl = axis.gridlines(crs = data_proj, color = "gray", alpha = 0.5, draw_labels = draw_labels,
                            linewidth = 0.5, linestyle = "dashed")

@dataclasses.dataclass
class xyCoords:
    x: str
    y: str

def determine_xy_coordinates(data_array):
    coords = data_array.coords   
 
    if ("lon" in coords) and ("lat" in coords):
        return xyCoords(x = "lon", y = "lat")
    elif ("latitude" in coords) and ("longitude" in coords):
        return xyCoords(x = "longitude", y = "latitude")
    elif ("x" in coords) and ("y" in coords):
        return xyCoords(x = "x", y = "y")
    elif ("grid_xt" in coords) and ("grid_yt" in coords):
        return xyCoords(x = "grid_xt", y = "grid_yt")
    else:
        print(f"Error: Unrecognized x,y coordinates in {coords} \nCan't proceed with plotting")
        sys.exit(1)

# Could be more sophisticated here, for example, by making
# sure the first dimension is a valid time dimension
def determine_if_has_time_dim(data_array):
    if (len(data_array.dims) == 3):
        return True
    return False

def how_to_plot_cmap_single_panel():
    print('plot_cmap_single_panel(data_array,\n'
          '                       data_name,\n'
          '                       region,\n'
          '                       plot_levels = np.arange(0, 85, 5),\n'
          '                       short_name = "precip_data",\n'
          '                       proj_name = "PlateCarree",\n'
          '                       cmap = DEFAULT_PRECIP_CMAP,\n'
          '                       extend = "max")')

# For each time in the data array, create a single-paneled contour plot of precipitation
def plot_cmap_single_panel(data_array,
                           data_name, region,
                           plot_levels = np.arange(0, 85, 5),
                           short_name = "precip_data",
                           proj_name = "PlateCarree",
                           cmap = DEFAULT_PRECIP_CMAP,
                           extend = "max"):
    match proj_name:
        case "LambertConformal":
            map_proj = ccrs.LambertConformal()
            data_proj = ccrs.PlateCarree()
        case _:
            map_proj = ccrs.PlateCarree()
            data_proj = ccrs.PlateCarree()
    
    # Handle dtime loop below according to whether a time dimension is present
    has_time_dim = determine_if_has_time_dim(data_array)
    if has_time_dim:
        time_dim = get_time_dimension_name(data_array)
        dtimes = [pd.Timestamp(i) for i in data_array[time_dim].values]
    else:
        time_dim = ""
        dtimes = [""]

    for dtime in dtimes:
        if (type(dtime) is pd.Timestamp):
            loc_str = dtime.strftime(utils.full_date_format_str) # Format is %Y-%m-%d %H:%M:%S
            dt_str = dtime.strftime("%Y%m%d.%H")
        elif (type(dtime) is str):
            loc_str = dtime
            dt_str = dtime
        else:
            print(f"Error: Invalid datetime {dtime} to select data; not continuing to make plots")
            return 

        # Select data to plot (at one valid time)
        if has_time_dim:
            data_to_plot = data_array.loc[loc_str]
        else:
            data_to_plot = data_array
            
        xy_coords = determine_xy_coordinates(data_array)

        # Set up the figure
        plt.figure(figsize = regions_info_dict[region].figsize_sp)
        axis = plt.axes(projection = map_proj)
        add_cartopy_features_to_map_proj(axis, region, data_proj, draw_labels = False)

        # Plot the data
        if (extend != "min") and (extend != "max") and (extend != "both"):
            extend = "both"
        plot_handle = data_to_plot.plot(ax = axis, levels = plot_levels, extend = extend, transform = data_proj, cmap = cmap,
                                        x = xy_coords.x, y = xy_coords.y, 
                                        cbar_kwargs = {"orientation": "horizontal", "ticks": plot_levels})


        # Configure color bar, axis labels and ticks, title, and figure name
        plot_handle.colorbar.set_label(data_array.units, size = 15) 
        plot_handle.colorbar.ax.set_xticklabels(plot_levels)
        plot_handle.colorbar.ax.tick_params(labelsize = 15)
        plt.xlabel("Latitude", fontsize = 15)
        plt.ylabel("Longitude", fontsize = 15)
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        
        # Determine short_name, a short descriptor of the data to use in plot title and figure name
        # Use the keyword arg first, otherwise try to get the short name from the data array attributes.
        if (type(short_name) is str) and (short_name != ""):
            formatted_short_name = short_name
        else: 
            try:
                formatted_short_name = precip_data_processors.format_short_name(data_array)
            except AttributeError:
                formatted_short_name = "precip_data" 
      
        # Create plot title 
        title_string = f"{data_name} {region} {formatted_short_name}" 
        if (type(dtime) is pd.Timestamp):
            title_string += f": valid at {dt_str}"
        elif (type(dtime) is str) and (has_time_dim):
            title_string += f": {dt_str}"
        plt.title(title_string, fontsize = 15, fontweight = "bold")
        plt.tight_layout()

        # Save figure
        if has_time_dim:
            fig_full_name = f"cmap.{data_name}.{formatted_short_name}.{dt_str}.{region}.png"
        else:
            fig_full_name = f"cmap.{data_name}.{formatted_short_name}.{region}.png"
        fig_path = os.path.join(utils.plot_output_dir, fig_full_name)
        print(f"Saving {fig_path}")
        plt.savefig(fig_path)

def how_to_plot_cmap_multi_panel():
    print('plot_cmap_multi_panel(data_dict,\n'
          '                      truth_data_name,\n'
          '                      region,\n'
          '                      figsize = None,\n'
          '                      plot_levels = np.arange(0, 85, 5),\n'
          '                      short_name = "precip_data",\n' 
          '                      sparse_cbar_ticks = False,\n'
          '                      cmap = DEFAULT_PRECIP_CMAP,\n'
          '                      extend = "max")')

# Contour maps with the correct number of panels, with the "truth" dataset always in the top left
def plot_cmap_multi_panel(data_dict,
                          truth_data_name,
                          region,
                          figsize = None,
                          plot_levels = np.arange(0, 85, 5),
                          short_name = "precip_data",
                          sparse_cbar_ticks = False,
                          cmap = DEFAULT_PRECIP_CMAP,
                          extend = "max"):
    # Configure basic info about the data
    truth_da = data_dict[truth_data_name]
    num_da = len(data_dict.items())
    data_names_str = get_data_names_str(data_dict) 
    if figsize is None:
        figsize = set_figsize_based_on_num_da(num_da, regions_info_dict[region])

    # Set map projection to be used for all subplots in the figure 
    proj = ccrs.PlateCarree()

    # Handle dtime loop below according to whether a time dimension is present
    has_time_dim = determine_if_has_time_dim(truth_da)
    if has_time_dim:
        time_dim = get_time_dimension_name(truth_da)
        dtimes = [pd.Timestamp(i) for i in truth_da[time_dim].values]
    else:
        time_dim = ""
        dtimes = [""]

    # Loop through these datetimes, making figures with subplots corresponding to each of the data arrays in data_dict
    for dtime in dtimes:
        if (type(dtime) is pd.Timestamp):
            loc_str = dtime.strftime(utils.full_date_format_str) # Format is %Y-%m-%d %H:%M:%S
            dt_str = dtime.strftime("%Y%m%d.%H")
        elif (type(dtime) is str):
            loc_str = dtime
            dt_str = dtime
        else:
            print(f"Error: Invalid datetime {dtime} to select data; not continuing to make plots")
            return 

        fig = plt.figure(figsize = figsize)
        axes_list = create_gridded_subplots(num_da, proj, layout = "grid")
        if (extend != "min") and (extend != "max") and (extend != "both"):
            extend = "both"
  
        # Loop through each of the subplot axes defined above (one axis for each DataArray) and plot the data 
        for axis, (data_name, da) in zip(axes_list, data_dict.items()):
            add_cartopy_features_to_map_proj(axis, region, proj)

            if has_time_dim:
                data_to_plot = da.loc[loc_str]
            else:
                data_to_plot = da

            xy_coords = determine_xy_coordinates(da)

            # Plot data 
            plot_handle = data_to_plot.plot(ax = axis,
                                            levels = plot_levels, 
                                            transform = proj, 
                                            extend = extend, 
                                            cmap = cmap,
                                            x = xy_coords.x, 
                                            y = xy_coords.y, 
                                            add_colorbar = False)

            # Configure colorbar
            # Colorbars with extend = "max" are very annoyingly off-center, too far to the left.
            # So, move them to the right a bit via axes positioning (items in list passed to 
            # add_axes method are [left, bottom, width, height]
            if (extend == "max"):
                cbar_ax = fig.add_axes([0.275, 0.1, 0.5, 0.02])
            else:
                cbar_ax = fig.add_axes([0.25, 0.1, 0.5, 0.02])
            cbar = fig.colorbar(plot_handle, cax = cbar_ax, ticks = plot_levels, shrink = 0.5, orientation = "horizontal")
            cbar.set_label(da.units, size = 15)
            #cbar_tick_labels_rotation, cbar_tick_labels_fontsize = set_cbar_labels_rotation_and_fontsize(plot_levels, region, num_da, for_single_cbar = True)
            cbar_tick_labels_rotation, cbar_tick_labels_fontsize = 45, 12
            cbar.ax.set_xticklabels(plot_levels, rotation = cbar_tick_labels_rotation) 
            cbar.ax.tick_params(labelsize = cbar_tick_labels_fontsize)
            axis.set_title(data_name, fontsize = 16)

        # Determine short_name, a short descriptor of the data to use in plot title and figure name
        # Use the keyword arg first, otherwise try to get the short name from the data array attributes.
        if (type(short_name) is str) and (short_name != ""):
            formatted_short_name = short_name
        else: 
            try:
                formatted_short_name = precip_data_processors.format_short_name(truth_da)
            except AttributeError:
                formatted_short_name = "precip_data" 

        # Create plot title
        if (type(dtime) is pd.Timestamp):
            title_string = f"{region} {formatted_short_name} valid at {dt_str}"
        elif (type(dtime) is str) and (has_time_dim):
            title_string = f"{region} {formatted_short_name}: {dt_str}"
        else:
            title_string = f"{region} {formatted_short_name}"
        fig.suptitle(title_string, fontsize = 16, fontweight = "bold")
        #fig.tight_layout()

        # Save figure
        if has_time_dim:
            fig_name = f"cmap.{data_names_str}{formatted_short_name}.{dt_str}.{region}.png"
        else:
            fig_name = f"cmap.{data_names_str}{formatted_short_name}.{region}.png"
        fig_path = os.path.join(utils.plot_output_dir, fig_name)
        print(f"Saving {fig_path}")
        plt.savefig(fig_path, bbox_inches = "tight")
  
def how_to_plot_blank_map_of_each_region():
    print("plot_blank_map_of_each_region()\n")
    print("Call it straight up, bro! Ain't no arguments required!")
    print("It will plot ALL the regions defined in regions_info_dict.")

# Plot a blank map for each region defined in regions_info_dict 
# This can be useful to assess whether the bounds for each region need to be adjusted, for example. 
def plot_blank_map_of_each_region():
    for region, region_config in regions_info_dict.items():
        plt.figure(figsize = region_config.figsize_sp)
        proj = ccrs.PlateCarree() 
        axis = plt.axes(projection = proj)
        add_cartopy_features_to_map_proj(axis, region, proj)
        plt.title(region, fontsize = 20)
        plt.tight_layout()

        fig_name = f"region.{region}.blank_map.png"
        fig_path = os.path.join(utils.plot_output_dir, fig_name)
        print(f"Saving {fig_path}") 
        plt.savefig(fig_path)
