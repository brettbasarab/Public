#!/usr/bin/env python

import matplotlib.pyplot as plt
import planetary_computer
import pystac_client
import xarray as xr

# Program to read CONUS404 data in zarr format.
# Zarr is a format optimized for storing multi-dimensional geospatial data in the cloud.
# See more here: wiki.earthdata.nasa.gov/display/ESO/Zarr+Format

#### PART 1: READ DATA FROM MICROSOFT'S PLANETARY COMPUTER (https://planetarycomputer.microsoft.com/) 
# Initial configuration to be able to interface with Planetary Computer
catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                    modifier = planetary_computer.sign_inplace)
c = catalog.get_collection("conus404")
asset = c.assets["zarr-abfs"]

# Read all CONUS404 data (all variables, all 40 years) as xarray dataset
conus404_ds = xr.open_zarr(asset.href,
                           storage_options = asset.extra_fields["xarray:storage_options"],
                           **asset.extra_fields["xarray:open_kwargs"],
                           )

# List all the variables within the Dataset
#conus404_ds.variables

#### PART 2: GET AND PLOT A PRECIPITATION TIME SERIES
# Get just precipitation variable as an xarray DataArray. 
accum_precip = conus404_ds["PREC_ACC_NC"]

# Select slices of data you want using this (overly?) intuitive .loc syntax.
# Nice not to have to find the numerical indices, as you would for numpy arrays.
# See docs.xarray.dev/en/stable/user-guide/indexing.html for more on xarray indexing.
# Here we get data over the CONUS for a week during September 2013, during the Colorado floods.
sep2013_data = accum_precip.loc["2013-09-08 01:00:00":"2013-09-15 00:00:00"]

# Get September 2013 data at a point near Boulder (i.e., a time series) by using the .sel syntax 
boulder_sep2013_data = sep2013_data.sel(south_north = 539, west_east = 529)

# Finally, if you want to get the data as a numpy array:
boulder_array = boulder_sep2013_data.values

# Plot the timeseries 
plt.figure(figsize = (10, 8))
plt.grid(True, linewidth = 0.5)
boulder_sep2013_data.plot()
plt.savefig("./test_conus404_timeseries.png")

