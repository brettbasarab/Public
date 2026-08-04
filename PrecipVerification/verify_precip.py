#!/usr/bin/env python

import argparse
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os
import precip_plotting_utilities as ppu
import precip_verification_processor
import sys
import utilities as utils
import warnings
import xarray as xr
import yaml

def main():
    utils.suppress_warnings()
    program_description = ("Program to create an instance of PrecipVerificationProcessor and use its functionality to create\n"
                          "many different types of plots, summarized in the flag descriptions below.")
    parser = argparse.ArgumentParser(description = program_description)
    parser.add_argument("yaml_fpath", 
                        help = "Path to yaml file containing verification configuration")
    args = parser.parse_args()

    # Parse yaml file
    if not(os.path.exists(args.yaml_fpath)):
        print(f"Error: yaml file path {args.yaml_fpath} does not exist")
        sys.exit(1)
    with open(args.yaml_fpath, "r") as f:
        yaml_data = yaml.load(f, Loader = yaml.FullLoader)

    # Input/output directories:
    try:
        input_dir = yaml_data["input_dir"]
    except KeyError:
        input_dir = None
    
    try:
        output_dir = yaml_data["output_dir"]
    except KeyError:
        output_dir = None

    try:
        user_dir = yaml_data["user_dir"]
    except KeyError:
        user_dir = "bbasarab"

    # Core configuration
    grid_cell_size = yaml_data["grid_cell_size"]

    # Stats configuration
    eval_radius_list_grid_cells = yaml_data["stats_config"]["eval_radius_list_grid_cells"]
    eval_radius_list_degrees = np.array(eval_radius_list_grid_cells) * grid_cell_size
    eval_threshold_list_ari = yaml_data["stats_config"]["eval_threshold_list_ari"]
    eval_threshold_list_mm = yaml_data["stats_config"]["eval_threshold_list_mm"]
    eval_threshold_list_pctl = yaml_data["stats_config"]["eval_threshold_list_pctl"]
    fixed_eval_radius_grid_cells = yaml_data["stats_config"]["fixed_eval_radius_grid_cells"]
    fixed_eval_radius_degrees = fixed_eval_radius_grid_cells * grid_cell_size
    fixed_eval_threshold_ari = yaml_data["stats_config"]["fixed_eval_threshold_ari"]
    fixed_eval_threshold_mm = yaml_data["stats_config"]["fixed_eval_threshold_mm"]
    fixed_eval_threshold_pctl = yaml_data["stats_config"]["fixed_eval_threshold_pctl"]
    fss_pctl_over_eval_period = yaml_data["stats_config"]["fss_pctl_over_eval_period"]
    time_period_types = yaml_data["stats_config"]["time_period_types"]
    timeseries_ann = yaml_data["stats_config"]["timeseries_ann"]

    # Which stats to calculate and plot
    do_cmaps = yaml_data["do_stats"]["cmaps"]
    do_fss = yaml_data["do_stats"]["fss"]
    do_fss_ari = yaml_data["do_stats"]["fss_ari"]
    do_fss_pctl = yaml_data["do_stats"]["fss_pctl"]
    do_fss_timeseries = yaml_data["do_stats"]["fss_timeseries"]
    do_occ_stats = yaml_data["do_stats"]["occ_stats"]
    do_pdfs = yaml_data["do_stats"]["pdfs"]
    do_scatter_plot = yaml_data["do_stats"]["scatter_plot"]
    do_mean_timeseries = yaml_data["do_stats"]["mean_timeseries"]

    # Write to nc configuration
    write_to_nc = yaml_data["write_to_nc"]

    regions_list = list(yaml_data["regions_info"].keys())
    for r, region in enumerate(regions_list):
        LOAD_DATA = True
        loaded_non_subset_da_dict = None 
        if (r > 0):
            LOAD_DATA = False
            loaded_non_subset_da_dict = verif.loaded_non_subset_da_dict # From previous instantiation of PrecipVerificationProcessor

        verif = precip_verification_processor.PrecipVerificationProcessor(yaml_data["start_dt_str"], 
                                                                          yaml_data["end_dt_str"],
                                                                          LOAD_DATA = LOAD_DATA, 
                                                                          loaded_non_subset_da_dict = loaded_non_subset_da_dict,
                                                                          USE_EXTERNAL_DA_DICT = yaml_data["USE_EXTERNAL_DA_DICT"],
                                                                          LAT_LON_2D = yaml_data["LAT_LON_2D"], 
                                                                          external_da_dict = None, 
                                                                          data_names = yaml_data["data_names"],
                                                                          truth_data_name = yaml_data["truth_data_name"],
                                                                          data_grid = yaml_data["data_grid"],
                                                                          grid_cell_size = grid_cell_size, 
                                                                          input_format = yaml_data["input_format"], 
                                                                          region = region,
                                                                          region_info = yaml_data["regions_info"][region],
                                                                          temporal_res = yaml_data["temporal_res"],
                                                                          poster = yaml_data["poster"],
                                                                          input_dir = input_dir,
                                                                          output_dir = output_dir,
                                                                          user_dir = user_dir) 

        # Calculate data for and plot contour maps of specified statistics valid at each grid point (so
        # contour maps can be made of the resulting data) and across various aggregation time periods.
        if (do_cmaps == "mean") or (do_cmaps == "all"):
            for time_period_type in time_period_types: 
                print(f"**** Calculating {time_period_type} mean data for contour maps")
                # Calculate mean
                agg_dict = verif.calculate_aggregated_stats(time_period_type = time_period_type,
                                                            stat_type = "mean",
                                                            agg_type = "time",
                                                            write_to_nc = write_to_nc)
                # Plot mean
                verif.plot_cmap_multi_panel(agg_dict,
                                            plot_levels = verif.region_plot_config.cm_mean_precip_range,
                                            plot_color_map = "terrain_r",
                                            extend = "max",
                                            sparse_cbar_ticks = True)
               
                # Plot errors of mean 
                verif.plot_cmap_multi_panel_errors(agg_dict,
                                                   plot_levels = verif.region_plot_config.cm_mean_precip_range,
                                                   plot_color_map = "terrain_r",
                                                   extend = "max",
                                                   sparse_cbar_ticks = True)
        if (do_cmaps == "pctls") or (do_cmaps == "all"):
            # FIXME: Update plot_levels for all percentile plots 
            for time_period_type in time_period_types:
                print(f"**** Calculating {time_period_type} percentile data for contour maps")
                # Calculate 95th percentile
                agg_dict = verif.calculate_aggregated_stats(time_period_type = time_period_type,
                                                            stat_type = "pctl",
                                                            agg_type = "time",
                                                            pctl = 95,
                                                            write_to_nc = write_to_nc) 
                # Plot 95th percentile 
                verif.plot_cmap_multi_panel(agg_dict,
                                            plot_levels = verif.region_plot_config.cm_mean_precip_range,
                                            plot_color_map = "terrain_r",
                                            extend = "max",
                                            sparse_cbar_ticks = True)
               
                # Plot errors of 95th percentile 
                verif.plot_cmap_multi_panel_errors(agg_dict,
                                                   plot_levels = verif.region_plot_config.cm_mean_precip_range,
                                                   plot_color_map = "terrain_r",
                                                   extend = "max",
                                                   sparse_cbar_ticks = True)

                # Calculate 99th percentile
                agg_dict = verif.calculate_aggregated_stats(time_period_type = time_period_type,
                                                            stat_type = "pctl",
                                                            agg_type = "time",
                                                            pctl = 99,
                                                            write_to_nc = write_to_nc)
                # Plot 99th percentile 
                verif.plot_cmap_multi_panel(agg_dict,
                                            plot_levels = verif.region_plot_config.cm_mean_precip_range,
                                            plot_color_map = "terrain_r",
                                            extend = "max",
                                            sparse_cbar_ticks = True)
               
                # Plot errors of 99th percentile 
                verif.plot_cmap_multi_panel_errors(agg_dict,
                                                   plot_levels = verif.region_plot_config.cm_mean_precip_range,
                                                   plot_color_map = "terrain_r",
                                                   extend = "max",
                                                   sparse_cbar_ticks = True)

                # Calculate 99.9th percentile
                agg_dict = verif.calculate_aggregated_stats(time_period_type = time_period_type,
                                                            stat_type = "pctl",
                                                            agg_type = "time",
                                                            pctl = 99.9,
                                                            write_to_nc = write_to_nc) 
                # Plot 99.9th percentile 
                verif.plot_cmap_multi_panel(agg_dict,
                                            plot_levels = verif.region_plot_config.cm_mean_precip_range,
                                            plot_color_map = "terrain_r",
                                            extend = "max",
                                            sparse_cbar_ticks = True)
               
                # Plot errors of 99.9th percentile 
                verif.plot_cmap_multi_panel_errors(agg_dict,
                                                   plot_levels = verif.region_plot_config.cm_mean_precip_range,
                                                   plot_color_map = "terrain_r",
                                                   extend = "max",
                                                   sparse_cbar_ticks = True)
        
        # Calculate and plot fractions skill score (FSS) by radius and amount threshold
        if do_fss: 
            print("******** Calculating FSS (using amount thresholds)")
            eval_threshold_list = utils.default_eval_threshold_list_mm

            # Calculate FSS by radius, using a fixed amount threshold 
            fss_dict_by_radius = verif.calculate_fss(eval_type = "by_radius",
                                                     fixed_threshold = fixed_eval_threshold_mm, 
                                                     eval_radius_list = eval_radius_list_degrees, 
                                                     is_pctl_threshold = False, 
                                                     include_zeros = False,
                                                     write_to_nc = write_to_nc) 

            # Calculate FSS by amount threshold, using a fixed evaluation radius
            fss_dict_by_thresh = verif.calculate_fss(eval_type = "by_threshold",
                                                     fixed_radius = fixed_eval_radius_degrees, 
                                                     eval_threshold_list = eval_threshold_list_mm,
                                                     is_pctl_threshold = False,
                                                     include_zeros = False,
                                                     write_to_nc = write_to_nc)
 
            # Plot FSS timeseries (using amount thresholds) for each time period across evaluation period
            if do_fss_timeseries: 
                print(f"Plotting FSS timeseries for eval radius {fixed_eval_radius_grid_cells*grid_cell_size:0.2f} deg; threshold {fixed_eval_threshold_mm:0.2f}mm")
                verif.plot_fss_timeseries(fixed_eval_radius_degrees, fixed_eval_threshold_mm) 
     
            # Plot FSS (using amount thresholds), averaged across evaluation period 
            for time_period_type in time_period_types: 
                print(f"**** Calculating and plotting {time_period_type} aggregated FSS (using amount thresholds)")
                verif.plot_aggregated_fss(eval_type = "by_radius",
                                          xaxis_explicit_values = False,
                                          time_period_type = time_period_type,
                                          is_pctl_threshold = False)

                verif.plot_aggregated_fss(eval_type = "by_threshold",
                                          xaxis_explicit_values = False,
                                          time_period_type = time_period_type,
                                          is_pctl_threshold = False, 
                                          include_frequency_bias = True)

        # Calculate and plot fractions skill score (FSS) by radius and percentile threshold 
        if do_fss_pctl: 
            print("******** Calculating FSS (using percentile thresholds)")
            eval_threshold_list = utils.default_eval_threshold_list_pctl

            # Calculate FSS by radius, using a fixed amount threshold 
            fss_dict_by_radius = verif.calculate_fss(eval_type = "by_radius",
                                                     fixed_threshold = fixed_eval_threshold_pctl, 
                                                     eval_radius_list = eval_radius_list_degrees, 
                                                     is_pctl_threshold = True, 
                                                     include_zeros = False, # whether to include zeros in percentile calculations
                                                     pctl_over_eval_period = fss_pctl_over_eval_period, 
                                                     write_to_nc = write_to_nc) 

            # Calculate FSS by amount threshold, using a fixed evaluation radius
            fss_dict_by_thresh = verif.calculate_fss(eval_type = "by_threshold",
                                                     fixed_radius = fixed_eval_radius_degrees, 
                                                     eval_threshold_list = eval_threshold_list_pctl,
                                                     is_pctl_threshold = True,
                                                     include_zeros = False, # whether to include zeros in percentile calculations
                                                     pctl_over_eval_period = fss_pctl_over_eval_period, 
                                                     write_to_nc = write_to_nc) 
            
            # Plot FSS timeseries (using amount thresholds) for each time period across evaluation period
            if do_fss_timeseries: 
                print(f"Plotting FSS timeseries for eval radius {fixed_eval_radius_grid_cells*grid_cell_size:0.2f} deg; threshold {fixed_eval_threshold_mm:0.2f}mm")
                verif.plot_fss_timeseries(fixed_eval_radius_degrees, fixed_eval_threshold_pctl)

            # Plot FSS (using percentile thresholds), averaged across evaluation period 
            for time_period_type in time_period_types: 
                print(f"**** Calculating and plotting {time_period_type} aggregated FSS (using percentile thresholds)")
                verif.plot_aggregated_fss(eval_type = "by_radius", xaxis_explicit_values = False,
                                          time_period_type = time_period_type,
                                          is_pctl_threshold = True)

                verif.plot_aggregated_fss(eval_type = "by_threshold", xaxis_explicit_values = False,
                                          time_period_type = time_period_type,
                                          is_pctl_threshold = True, 
                                          include_frequency_bias = True)

        # Calculate and plot fractions skill score (FSS) by radius and ARI grid threshold 
        if do_fss_ari: 
            print("******** Calculating FSS (using ARI grid thresholds)")

            # Calculate FSS by radius, using an ARI grid threshold
            fss_dict_by_radius = verif.calculate_fss(eval_type = "by_radius_ari_threshold",
                                                     fixed_ari_threshold = fixed_eval_threshold_ari, 
                                                     eval_radius_list = eval_radius_list_degrees,
                                                     write_to_nc = write_to_nc) 

            # Calculate FSS by ARI grid (using varying ARI grids thresholds) 
            fss_dict_by_ari = verif.calculate_fss(eval_type = "by_ari_grid",
                                                  fixed_radius = fixed_eval_radius_degrees, 
                                                  eval_ari_list = eval_threshold_list_ari, 
                                                  write_to_nc = write_to_nc) 

            # Plot FSS (using ARI grid thresholds), averaged across evaluation period 
            for time_period_type in time_period_types: 
                print(f"**** Calculating and plotting {time_period_type} aggregated FSS (using ARI grid thresholds)")
                verif.plot_aggregated_fss(eval_type = "by_radius_ari_threshold", xaxis_explicit_values = False,
                                          time_period_type = time_period_type)
                verif.plot_aggregated_fss(eval_type = "by_ari_grid", xaxis_explicit_values = True,
                                          time_period_type = time_period_type)
    
        # Calculate and plot occurrence stats (CSI, etc.)
        if do_occ_stats: 
            for time_period_type in time_period_types: 
                print(f"**** Calculating {time_period_type} occurrence statistics") 
                occ_stats_dict = verif.calculate_aggregated_occ_stats(which_stat = "all",
                                                                      time_period_type = time_period_type,
                                                                      write_to_nc = write_to_nc) 
                verif.plot_aggregated_occ_stats_by_threshold(occ_stats_dict, which_stat = "CSI", time_period_type = time_period_type) 
                verif.plot_aggregated_occ_stats_by_threshold(occ_stats_dict, which_stat = "ETS", time_period_type = time_period_type) 
                verif.plot_aggregated_occ_stats_by_threshold(occ_stats_dict, which_stat = "frequency_bias", time_period_type = time_period_type) 

        # Plot PDFs and CDFs
        if do_pdfs: 
            for time_period_type in time_period_types: 
                print(f"**** Calculating {time_period_type} PDFs and CDFs")
                pdf_dict = verif.calculate_pdf(time_period_type = time_period_type, write_to_nc = write_to_nc) 
                verif.plot_pdf(data_dict = pdf_dict, time_period_type = time_period_type) 

        # Plot monthly and seasonal mean timeseries
        if do_mean_timeseries: 
            for time_period_type in ["common_monthly", "seasonal"]:
                print(f"**** Calculating {time_period_type} timeseries stats")
                agg_dict = verif.calculate_aggregated_stats(time_period_type = time_period_type,
                                                            stat_type = "mean",
                                                            agg_type = "space_time",
                                                            write_to_nc = write_to_nc) 
                verif.plot_timeseries(data_dict = agg_dict,
                                      time_period_type = time_period_type,
                                      stat_type = "mean",
                                      plot_levels = verif.region_plot_config.ts_mean_precip_range,
                                      ann_plot = timeseries_ann)

        # Plot scatter plot of accumulated precip over entire time period
        if do_scatter_plot: 
            print("**** Creating scatter plots of accumulated precip summed over entire time period")
            verif.plot_scatter_plot(summed_over_time_period = True)

    return verif

if __name__ == "__main__":
    verif = main() 
