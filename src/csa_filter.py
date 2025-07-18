"""
Take cross-spectral analysis results produced by csa_multiprocess.py and compute average coherency across
a specified period band, testing for significance based on the large-scale neighbourhood of each pixel.
Proceed to compute the average period at which significant coherency occurs, the average phase difference
and the width of the 95% confidence interval for the phase difference.

Works with the same four latitude bands as csa_multiprocess.py.

Saves maps of neighbourhood average properties in a dictionary using pickle.

Bethan Harris, UKCEH, 18/11/2020
"""

import numpy as np
import os
import time
import sys
import pickle
import itertools
from multiprocessing import Pool

from read_csagan_saved_output import read_region_data
import utils.load as ul


_global_shared_data = {}


def neighbourhood_indices(lat_idx, lon_idx):
    """
    Get indices for all neighbouring pixels 
    based on centre coordinates (see Harris et al. 2022, Fig. S2).
    Parameters
    ----------
    lat_idx: int
        Index representing position of central pixel on axis 0.
    lon_idx: int
        Index representing position of central pixel on axis 1.
    Returns
    -------
    all_pixels: itertools.product object
        Iterable of tuples (lat_idx, lon_idx) for all neighbouring pixels.
        Includes the central pixel itself.
    """
    lat_idcs = range(lat_idx-4, lat_idx+5, 4)
    lon_idcs = range(lon_idx-4, lon_idx+5, 4)
    all_pixels = itertools.product(lat_idcs, lon_idcs)
    return all_pixels


def neighbourhood_spectra(spectra_data, lat_idx, lon_idx):
    """Extract spectral period and coherency data from neighbouring pixels.

    Turn the array of dictionaries of cross-spectral analysis output into lists
    of arrays of period and coherency for each neighbouring pixel plus central
    pixel.  Each list has one item per pixel in neighbourhood.  Neighbouring
    pixels are not necessarily adjacent to the central pixel, see
    neighbourhood_indices() for exact definition.

    Index names make sense if array has latitude as axis 0, longitude as axis 1
    (both increasing with axis index).

    Parameters
    ----------
    spectra_data: numpy array of dicts
        Array of dictionaries from cross-spectral analysis output
    lat_idx: int
        Index representing position of central pixel on axis 0.
    lon_idx: int
        Index representing position of central pixel on axis 1.
    Returns
    -------
    periods: list of 1D arrays
        Each element in the list is an array of the periods sampled at either
        the central pixel or one of its neighbours.
    coherencies: list of 1D arrays
        Each element in the list is an array of the coherencies computed at either
        the central pixel or one of its neighbours, for the corresponding periods
        from list_of_periods.

    """
    periods = []
    coherencies = []
    for pixel_lat, pixel_lon in neighbourhood_indices(lat_idx, lon_idx):
        in_bounds = ( (0 <= pixel_lat < spectra_data.shape[0]) and
                      (0 <= pixel_lon < spectra_data.shape[1]) )
        is_central_pixel = (pixel_lat == lat_idx) and (pixel_lon == lon_idx)
        if in_bounds and not is_central_pixel:
            spectrum = spectra_data[pixel_lat, pixel_lon]
            # Only interested in pixels that have data from the cross-spectral
            # analysis. Return reverse data so periods are increasing.
            if spectrum != {}:
                periods.append(spectrum['period'][::-1])
                coherencies.append(spectrum['coherency'][::-1])

    return periods, coherencies


def check_significant_neighbours(spectra_data, lat_idx, lon_idx, band_days_lower, band_days_upper):
    """
    Find periods of variability that show significant coherency (95% CL) in the central pixel
    and also in at least 3 neighbouring pixels. Return the periods, phase differences,
    coherencies and amplitude ratios at the periods that show this significant coherency.
    Parameters
    ----------
    spectra_data: numpy array of dicts
        Array of dictionaries from cross-spectral analysis output
    lat_idx: int
        Index representing position of central pixel on axis 0.
    lon_idx: int
        Index representing position of central pixel on axis 1.
    Returns
    -------
    None if the central pixel has no output from cross-spectral analysis.
    Otherwise:
    central_periods: 1D array
        Periods within the intraseasonal band (defined in configuration section
        at top of script) which exhibit significant coherency at the 95% confidence
        level for the central pixel and pass the neighbour-based robustness test.
    central_phases: 1D array
        Phase difference (for central pixel) at the periods given by central_periods.
    central_coherencies: 1D array
        Coherency (for central pixel) at the periods given by central_periods.
    central_amplitudes: 1D array
        Amplitude ratio (for central pixel) at the periods given by central_periods.       
    """
    central_spectra = spectra_data[lat_idx, lon_idx]
    if central_spectra != {}:
        central_periods = central_spectra['period'][::-1]
        central_coherencies = central_spectra['coherency'][::-1]

        neighbour_periods, neighbour_coherencies = neighbourhood_spectra(spectra_data, lat_idx, lon_idx)

        central_period_band = np.logical_and(central_periods<=band_days_upper, central_periods>=band_days_lower)

        # The following line will raise exception if len(central_period_band) <
        # 2, and no data will be produced for this pixel.  The return value
        # min_period_gap is never used, so is this being used as cryptic flow
        # control.
        min_period_gap = np.diff(central_periods[central_period_band]).min()
        central_sig_periods = central_coherencies > 0.7795
        central_sig_periods_in_band = central_periods[np.logical_and(central_period_band, central_sig_periods)]
        significant_neighbours = np.zeros_like(central_sig_periods_in_band)
        resolution_bandwidth = central_spectra['resolution_bandwidth']
        min_periods = 1./((1./central_sig_periods_in_band) + 0.5*resolution_bandwidth)
        max_periods = 1./((1./central_sig_periods_in_band) - 0.5*resolution_bandwidth)

        for p, c in zip(neighbour_periods, neighbour_coherencies):
            for i, test_period in enumerate(central_sig_periods_in_band):
                period_within_rbw = np.logical_and(p<=max_periods[i], p>=min_periods[i])
                coh_sig_within_rbw = np.logical_and(period_within_rbw, c>0.7795)
                significant_neighbours[i] += int(np.any(coh_sig_within_rbw))

        significant_periods = central_sig_periods_in_band[significant_neighbours>2.]
        significant_idcs = np.isin(central_periods, significant_periods)

        central_phases = central_spectra['phase'][::-1]
        central_amplitudes = central_spectra['amplitude'][::-1]

        return (central_periods[significant_idcs],
                central_phases[significant_idcs],
                central_coherencies[significant_idcs],
                central_amplitudes[significant_idcs])
    else: # no cross-spectral output (likely due to insufficient obs, e.g. over ocean for VOD)
        return None


def compute_phase_error(coherency):
    """
    Calculate error (half width of 95% confidence interval) in phase difference in degrees
    from coherency. Adapted to python from csagan.f code.
    Parameters
    ----------
    coherency: float or array of floats
    Returns
    -------
    final_errors_degrees: same dtype as coherency
    """
    # Check whether coherency has been supplied as one float value only (rather than array)
    return_float = False
    if isinstance(coherency, float):
        coherency = np.array([coherency])
        return_float = True
    error_coeff = 0.238 / (2. * (1. - 0.238))
    phase_error = np.sqrt(error_coeff * (1./coherency**2 - 1.))
    final_errors = np.empty_like(phase_error)
    within_90 = (2.447 * phase_error) < 1.
    final_errors[within_90] = np.arcsin(2.447 * phase_error[within_90])
    final_errors[~within_90] = np.maximum(np.arcsin(1.), 1.96*np.sqrt(0.238)*np.sqrt(0.5 * (1./coherency[~within_90]**2 - 1.)))
    final_errors_degrees = 180. * final_errors / np.pi
    if return_float:
        return final_errors_degrees[0]
    else:
        return final_errors_degrees


def average_intraseasonal_coherency(coords):
    """
    Get average properties from cross-spectral analysis
    across intraseasonal frequency band for a single pixel (includes
    neighbour-based robustness test on coherency for each frequency).
    Parameters
    ----------
    coords: tuple (int, int)
        Indices of pixel (lat_idx, lon_idx) in the loaded tile of cross-spectral
        analysis data.
    Returns
    -------
    avg_coherency: float
        Mean coherency across all frequencies in intraseasonal band where
        coherency is significant at 95% confidence level in pixel
        and passes neighbourhood robustness test.
    avg_period: float
        Mean of periods in intraseasonal band where coherency is significant.
    avg_phase: float
        Mean phase difference (in degrees) at frequencies in intraseasonal band 
        where coherency is significant.
    avg_lag: float
        Mean lag (in days) at frequencies in intraseasonal band 
        where coherency is significant.
    lag_error: float
        Error in intraseasonal band-mean lag, calculated from avg_coherency
    avg_amplitude: float
        Mean amplitude ratio at frequencies in intraseasonal band 
        where coherency is significant.
    """

    global _global_shared_data
    spectra = _global_shared_data["spectra"]
    band_days_lower = _global_shared_data["band_days_lower"]
    band_days_upper = _global_shared_data["band_days_upper"]

    lat, lon = coords
    if spectra[lat, lon] != {}: # check if any output from cross-spectral analysis at pixel
        try:
            nbhd_test = check_significant_neighbours(spectra, lat, lon, band_days_lower, band_days_upper)
            if nbhd_test is not None:
                sig_periods = nbhd_test[0]
                sig_phases = nbhd_test[1]
                sig_coherencies = nbhd_test[2]
                sig_amplitudes = nbhd_test[3]
                if sig_periods.size > 1:
                    # More than one period in intraseasonal band with significant coherency
                    # so take means across these periods
                    avg_coherency = np.mean(sig_coherencies)
                    avg_period = np.mean(sig_periods)
                    # For large phase differences, shift to [0, 360] degrees before
                    # averaging, then back to [-180, 180]
                    # (explained in Section 2.3 of Harris et al. 2022)
                    if np.any(np.abs(sig_phases) > 150.):
                        sig_phases[sig_phases<0.] += 360.
                    avg_phase = np.mean(sig_phases)
                    if avg_phase > 180.:
                        avg_phase -= 360.
                    # Need to use period-weighted averages to get values of average lag in
                    # days which are consistent with the average phase difference
                    # in degrees
                    weighted_avg_phase = np.average(sig_phases, weights=sig_periods)
                    weighted_avg_coh = np.average(sig_coherencies, weights=sig_periods)
                    avg_lag = weighted_avg_phase / 360. * avg_period
                    if avg_lag > 0.5*avg_period:
                        avg_lag -= avg_period
                    avg_amplitude = np.mean(sig_amplitudes)
                    phase_error = compute_phase_error(avg_coherency)
                    lag_error = compute_phase_error(weighted_avg_coh) / 360. * avg_period
                else:
                    # If only one period with significant coherency, just take values
                    # at that period
                    avg_coherency = sig_coherencies[0]
                    avg_period = sig_periods[0]
                    avg_phase = sig_phases[0]
                    avg_lag = avg_phase / 360. * avg_period
                    avg_amplitude = sig_amplitudes[0]
                    phase_error = compute_phase_error(avg_coherency)
                    lag_error = phase_error / 360. * avg_period
            else:
                avg_coherency = np.nan
                avg_period = np.nan
                avg_phase = np.nan
                avg_lag = np.nan
                avg_amplitude = np.nan
                lag_error = np.nan
        except (IndexError, ValueError):
            # These are raised if (1) sig_* components are not indexable, (2)
            # Pixels don't sample enough frequencies for the requested period
            # band.
            avg_coherency = np.nan
            avg_period = np.nan
            avg_phase = np.nan
            avg_lag = np.nan
            avg_amplitude = np.nan
            lag_error = np.nan
    else:
        avg_coherency = np.nan
        avg_period = np.nan
        avg_phase = np.nan
        avg_lag = np.nan
        avg_amplitude = np.nan
        lag_error = np.nan
    return avg_coherency, avg_period, avg_phase, avg_lag, lag_error, avg_amplitude


def run_neighbourhood_averaging(lats, lons):
    """
    Get average properties from cross-spectral analysis
    across intraseasonal frequency band for every pixel in tile (includes
    neighbour-based robustness test).
    Parameters:
    None
    Returns (dict):
    Dictionary containing 2D arrays of neighbourhood average coherency, period, 
    phase difference, lag, error in lag (half width of 95% CI) and amplitude ratio.
    """
    lat_idcs = np.arange(lats.size)
    lon_idcs = np.arange(lons.size)
    LAT, LON = np.meshgrid(lat_idcs, lon_idcs)
    coords = zip(LAT.ravel(), LON.ravel())
    neighbourhood_averages = {}
    print(f'Start averaging {lats.size * lons.size} pixels')

    start = time.time()
    with Pool(processes=4) as pool:
        output = pool.map(average_intraseasonal_coherency, coords, chunksize=1)

    output_array = np.array(output)
    neighbourhood_averages['coherency'] = np.reshape(output_array[:, 0], (lats.size, lons.size), order='F')
    neighbourhood_averages['period'] = np.reshape(output_array[:, 1], (lats.size, lons.size), order='F')
    neighbourhood_averages['phase'] = np.reshape(output_array[:, 2], (lats.size, lons.size), order='F')
    neighbourhood_averages['lag'] = np.reshape(output_array[:, 3], (lats.size, lons.size), order='F')
    neighbourhood_averages['lag_error'] = np.reshape(output_array[:, 4], (lats.size, lons.size), order='F')
    neighbourhood_averages['amplitude'] = np.reshape(output_array[:, 5], (lats.size, lons.size), order='F')
    end = time.time()
    print(f'Time taken to compute intraseasonal averages: {end-start:0.1f} seconds')
    return neighbourhood_averages


def run_all_processing(output_dirs, datasets, tiles, seasons, bands):
    spectra_dir = output_dirs["spectra"]
    output_dir = output_dirs["spectra_filtered"]

    reference_var = datasets["reference_var"]
    response_var = datasets["response_var"]

    for tile, tile_bounds in tiles.items():
        for season in seasons:
            spectra_file = os.path.join(
                spectra_dir,
                f"{tile}_{reference_var}_{response_var}_spectra_{season}_mask_sw_best85.pkl"
            )
            print(f"Reading {spectra_file}")

            lats, lons, spectra = read_region_data(
                spectra_file, tile, *tile_bounds, resolution=0.25
            )

            for (bl, bu) in bands:
                # Load data into the global dictionary for sharing across the
                # multiprocessing pool.
                global _global_shared_data
                _global_shared_data["spectra"] = spectra
                _global_shared_data["band_days_lower"] = bl
                _global_shared_data["band_days_upper"] = bu

                print(f"Processing band ({bl}-{bu})")
                neighbourhood_averages = run_neighbourhood_averaging(lats, lons)

                output_file = os.path.join(
                    output_dir,
                    f"spectra_nooverlap_{tile}_{reference_var}_{response_var}_{season}_sw_filter_best85_{bl}-{bu}.pkl"
                )
                print(f"Writing {output_file}")

                with open(output_file, 'wb') as f:
                    pickle.dump(neighbourhood_averages, f)

    return


def main():

    ###########################################################################
    # Parse command line args and load input file.
    ###########################################################################
    parser = ul.get_arg_parser()
    args = parser.parse_args()

    metadata = ul.load_yaml(args)

    output_dirs = metadata.get("output_dirs", None)

    datasets = metadata.get("datasets", None)

    bands = [tuple(b) for b in metadata["filter"].get("bands", None)]
    seasons = metadata["spectra"].get("seasons", None)
    tiles = metadata["spectra"].get("tiles", None)

    # Do some sanity checking of processed input args.
    for (bl, bu) in bands:
        if bl <= 0 or bu <= 0:
            sys.exit(f"ERROR: Band bounds are not positive: [{bl} to {bu}]")
        elif bl > bu:
            sys.exit(f"ERROR: Band bounds must be increasing: [{bl} to {bu}]")

    ul.check_dirs(output_dirs,
                  input_names=("spectra",),
                  output_names=("spectra_filtered",))

    # Run all requested cross-spectral analyses.
    run_all_processing(output_dirs, datasets, tiles, seasons, bands)

    return


if __name__ == '__main__':
    main()
