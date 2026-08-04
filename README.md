# Precip Verification Processor

## Input Files Important Points
* All data files must be on the same grid (this code does not handle interpolation/regridding).
* Only south->north dimension names 'lat', 'latitude', 'y', or 'south_north' are supported
* Only west->east dimension names 'lon', 'longitude', 'x', or 'west_east' are supported


## Running Basic Verification for Big Sioux River Case Study
* In verify_precip.TruthAORC.GridAORC.01.BigSioux.yaml, at the top edit:
```
input_dir: contains netCDF files
output_dir: locations of plots/ and stats/ output directories, the code will create these
user_dir: <first_initial><last_name> on PSL Linux servers, e.g., bbasarab
```

* Run the code in interactive mode; pass the yaml file as an argument to verify_precip.py
```
python -i verify_precip.py verify_precip.TruthAORC.GridAORC.01.BigSioux.yaml
```
As the yaml file is currently configured, no calculations or plots will actually be made. This is the idea for this quick tutorial, so a few examples can be run interactively (next).
