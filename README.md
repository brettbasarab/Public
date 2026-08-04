# Precip Verification Processor

## Input Files Important Points
* All data files must be on the same grid (this code does not handle interpolation/regridding).
* Only south->north dimension names 'lat', 'latitude', 'y', or 'south_north' are supported
* Only west->east dimension names 'lon', 'longitude', 'x', or 'west_east' are supported

## Running Basic Verification for Big Sioux River Case Study

### Instantiate PrecipVerificationProcessor class
In verify_precip.TruthAORC.GridAORC.01.BigSioux.yaml, at the top edit:
```
input_dir: contains netCDF files
output_dir: locations of plots/ and stats/ output directories, the code will create these
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
Note the useful helper methods with match the name of the actual method but start with how_to... These print a basic usage statement (list of arguments and some helpful notes).
```
verif.num_valid_times
sum_dict = verif.get_summed_data() # Calculate summed data
for data_name, da in sum_dict.items(): print(data_name, da.shape)
verif.how_to_plot_cmap_multi_panel() # Helper method
verif.plot_cmap_multi_panel(sum_dict) # Plot total accumulated precipitation
verif.how_to_plot_cmap_multi_panel_errors()
verif.plot_cmap_multi_panel_errors(sum_dict, cbar_tick_labels_rotation = 60, sparse_cbar_ticks = True) # Plot AORC total accumulated precipitation and errors of other datasets we're evaluating
```

### Calculate fractions skill score (FSS) of total accumulated precipitation
```
verif.how_to_calculate_fss()
verif.grid_cell_size
fss_dict = verif.calculate_fss(eval_type = "by_threshold", da_dict = sum_dict, fixed_radius = 12 * verif.grid_cell_size)
for data_name, da in fss_dict.items(): print(data_name, da.shape)
fss_dict["CONUS404"]
fss_dict["CONUS404"].threshold
verif.how_to_plot_aggregated_fss()
verif.plot_aggregated_fss(da_dict = sum_dict, eval_type = "by_threshold")
```

### Plot FSS timeseries of hourly precipitation
