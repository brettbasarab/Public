#!/usr/bin/env python

import math
import numpy as np
import scipy

# Set data array to 1 at or above threshold, zero below it
# NOTE: this function takes an xarray DataArray as input and returns a numpy array.
def mask_data_array_based_on_threshold(da, threshold):
    # Convert the arrays to 2-D, removing the time dimension
    if (len(da.shape) == 3):
        da = da[0,:,:]

    return np.where(da >= threshold, 1, 0) 

# Create circular footprint for FSS calculation
# Obtain a square of zeros (with size length, in grid squares, of radius/grid_cell_size * 2 + 1)
# circumscribing a circle of ones (with radius, in grid_squares, of radius/grid_cell_size)
def get_footprint(radius, grid_cell_size):
    radius_number_grid_cells = int(radius/grid_cell_size)


    # 1) Create a footprint: just an array of 1s
    footprint = (np.ones((radius_number_grid_cells * 2 + 1, radius_number_grid_cells * 2 + 1))).astype(int)

    # 2) Set the centerpoint of the array to zero (needed for the subsequent distance calculation)
    footprint[int(math.ceil(radius_number_grid_cells)), int(math.ceil(radius_number_grid_cells))] = 0

    # 3) Within the footprint, calculate each point's distance from the center point
    dist = scipy.ndimage.distance_transform_edt(footprint, sampling = [grid_cell_size, grid_cell_size])

    # 4) Set the footprint to zeros where distance calculated in step 3) is greater than radius; keep it
    # set to one where distance is less than radius, obtaining the square of zeros circumscribing the circle of ones 
    return np.where(np.greater(dist, radius), 0, 1)

# FSS calculation from Craig Schwartz via Trevor Alcott (NOAA)
def calculate_fss(qpf, qpe, radius, grid_cell_size, threshold):
    # Calculate footprint, i.e., evaluation area
    footprint = get_footprint(radius, grid_cell_size)

    # Convert qpf and qpe arrays to numpy arrays containing 1s and 0s based on
    # whether precipitation amount is (at or above) or below threshold   
    binary_qpf =  mask_data_array_based_on_threshold(qpf, threshold)
    binary_qpe = mask_data_array_based_on_threshold(qpe, threshold)

    # Calculate forecast_fractions and observed fractions terms in the FSS formula.
    # These are the M (model) and O (observed) terms calculated in Roberts and Lean (2008) Equations 2 and 3
    # (https://doi.org/10.1175/2007MWR2123.1); what Trevor's code refers to as pf and po, respectively.
    # CONCEPTUAL PROCEDURE:
        # For every grid point, calculate the number of points within the footprint centered on the grid point
        # for which the binary_qpf and binary_qpe arrays equal 1 (i.e., <qpf> and <qpe> are at or above <threshold>).
        # Divide by the size of the footprint [np.sum(footprint)] to convert this number of points to a spatial
        # fraction of the footprint.
    forecast_fractions = np.around(scipy.signal.fftconvolve(binary_qpf, footprint, mode = "same"))/np.sum(footprint)
    observed_fractions = np.around(scipy.signal.fftconvolve(binary_qpe, footprint, mode = "same"))/np.sum(footprint)

    # Calculate gridsize (Nx * Ny)
    gridsize = np.shape(qpe)[0] * np.shape(qpe)[1]

    # Calculate numerator [Equation 5 in Roberts and Lean (2008)]
    mse = 1/gridsize * np.sum((forecast_fractions - observed_fractions)**2)

    # Calculate denominator [Equation 7 in Roberts and Lean (2008)]
    mse_reference = 1/gridsize * (np.sum(forecast_fractions**2) + np.sum(observed_fractions**2))

    if (mse_reference > 0):
        return 1.0 - float(mse)/float(mse_reference)
    else:
        return np.nan 
