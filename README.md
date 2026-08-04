# Precip Verification Processor

## Getting started
Clone the repo and cd to the precipitation verification directory. You'll run the code directly in this directory.
```
git clone https://github.com/brettbasarab/Public.git
cd PrecipVerification
```

## Crucial Points Regarding the Input Files
* All data files must already be on the same grid (this code does not handle interpolation/regridding).
* Only south->north dimensions named 'lat', 'latitude', 'y', or 'south_north' are supported
* Only west->east dimensions named 'lon', 'longitude', 'x', or 'west_east' are supported

## Running Basic Verification for Big Sioux River Case Study

### Instantiate PrecipVerificationProcessor class
In verify_precip.TruthAORC.GridAORC.01.BigSioux.yaml, at the top edit:
```
input_dir: contains subdirectories (one for each dataset being used) which contain individual netCDF files
output_dir: desired location of plots/ and stats/ output subdirectories; the code will create these subdirectories if they don't yet exist.
user_dir: <first_initial><last_name> on PSL Linux servers, e.g., bbasarab
```

Run the code in interactive mode; pass the yaml file as an argument to verify_precip.py
```
python -i ./verify_precip.py ./verify_precip.TruthAORC.GridAORC.01.BigSioux.yaml
```
As the yaml file is currently configured, no calculations or plots will actually be made. This is the idea for this quick tutorial, so a few examples can be run interactively (next).

### See some basic attributes of the class
Copy and paste the following into the interactive Python interpreter:
```
verif
verif.data_names
verif.truth_data_name
print(verif.start_dt)
print(verif.end_dt)
verif.temporal_res
verif.valid_dt_list
```

### Calculate total accumulated precipitation over the 168-hour evaluation period and plot it
Note the useful 'how to' methods that match the name of the actual method but start with how_to... These print a basic usage statement (list of arguments and some helpful notes).
```
verif.num_valid_times
sum_dict = verif.get_summed_data() # Calculate total accumulated precipitation
for data_name, da in sum_dict.items(): print(data_name, da.shape)
verif.how_to_plot_cmap_multi_panel() # How-to method
verif.plot_cmap_multi_panel(sum_dict) # Plot total accumulated precipitation for all datasets on a multi-panel plot
verif.how_to_plot_cmap_multi_panel_errors() # How-to method
verif.plot_cmap_multi_panel_errors(sum_dict, cbar_tick_labels_rotation = 60, sparse_cbar_ticks = True) # Plot AORC total accumulated precipitation and errors of other datasets we're evaluating
```

### Calculate fractions skill score (FSS) of total accumulated precipitation
```
verif.how_to_calculate_fss()
verif.grid_cell_size
fss_dict = verif.calculate_fss(da_dict = sum_dict, eval_type = "by_threshold", da_dict = sum_dict, fixed_radius = 12 * verif.grid_cell_size)
for data_name, da in fss_dict.items(): print(data_name, da.shape)
fss_dict["CONUS404"]
fss_dict["CONUS404"].threshold
verif.how_to_plot_aggregated_fss()
verif.plot_aggregated_fss(da_dict = sum_dict, eval_type = "by_threshold")
```

### Plot FSS timeseries of hourly precipitation
Here, it's easier to run the verification again, but in the `do_stats` section of the yaml file, set `fss: True`. Setting this flag to true will calculate FSS at varying amount (mm) thresholds and evaluation neighborhoods (radii, in degrees). It will take a bit of time. This flag will also make plots of aggregated FSS (across the time period) by threshold and radius, although we're interested in the timeseries, which we'll plot below. Exit the Python interpreter and then run the following:
```
python -i ./verify_precip.py ./verify_precip.TruthAORC.GridAORC.01.BigSioux.yaml
```
Once in the Python interpreter again, run:
```
verif.how_to_plot_fss_timeseries()
verif.grid_cell_size
verif.plot_fss_timeseries(12 * verif.grid_cell_size, 10) # 0.4 degree evaluation radius, 10 mm threshold
verif.plot_fss_timeseries(12 * verif.grid_cell_size, 25) # 0.4 degree evaluation radius, 25 mm threshold
```
