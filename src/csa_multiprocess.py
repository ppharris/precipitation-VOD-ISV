import calendar
from cftime import num2date
from datetime import datetime
from multiprocessing import Pool, RawArray
from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma
import os
from pathlib import Path
import pickle
import subprocess
from subprocess import run
import sys
import time

from read_data_iris import read_data_all_years, read_land_sea_mask
from utils.datasets import get_dataset
from utils.datetime import decimal_year_to_datetime, days_since_1970_to_decimal_year
import utils.load as ul

# Perform cross-spectral analysis on 3D netCDF data using csagan and save results (with pickle).
# csagan is run separately for each pixel to allow for each pixel having missing data
# at different time steps. Multiprocessing is used to make this feasible.
# Analysis is performed in one of four latitude bands ("polar", northern", "tropics", "southern") 
# to fit data in memory. The latitude bands overlap slightly in order to make it easier
# to test the neighbours of each pixel later on in the processing chain.
# See the bash script csa_multiprocess_tiles to run this script for all latitude bands/seasons.
# read_csagan_saved_output.py will read in the pickle files that this script produces.
# Designed to work with 19 years of daily data at 0.25° horizontal resolution.


init_dict = {} # Dictionary to contain data that needs sharing between multiprocessing workers


def pad_match_time(dates1, dates2, data1, data2):
    """
    For two arrays of data with dimensions time/lon/lat or time/lat/lon,
    add grids filled with NaNs along 0th dimension to one or both arrays
    as appropriate so that they cover the same time stamps.
    Parameters
    ----------
    dates1: numpy array (1D)
        Dates corresponding to the observations in array data1
    dates2: numpy array (1D)
        Dates corresponding to the observations in array data2
    data1: numpy array (3D)
        Array of data observed at times from dates1
    data2: numpy array (3D)
        Array of data observed at times from dates2
    Returns
    -------
    dates: numpy array (1D)
        Array of dates covering all observations in both datasets
    data1_pad: numpy array (3D)
        Data from data1 matched to common timestamps of dates
    data2_pad: numpy array (3D)
        Data from data2 matched to common timestamps of dates
    """
    # Check whether the arrays already share the same time stamps
    if (dates1.shape == dates2.shape) and np.allclose(dates1, dates2):
    	return dates1, data1, data2
    else:
    	dates = np.union1d(dates1, dates2)
    	data1_times = np.isin(dates, dates1)
    	data2_times = np.isin(dates, dates2)
    	data1_pad = np.empty((dates.size, data1.shape[1], data1.shape[2]))
    	data2_pad = np.empty((dates.size, data2.shape[1], data2.shape[2]))
    	data1_pad[:] = np.nan
    	data2_pad[:] = np.nan
    	data1_pad[data1_times, :, :] = data1
    	data2_pad[data2_times, :, :] = data2
    	return dates, data1_pad, data2_pad


def season_from_abbr(season_abbr):
    """
    Produce a list of month numbers from a season abbreviation,
    e.g. MAM -> [3, 4, 5]
    Parameters
    ----------
    season_abbr: str
        Abbreviation for season (i.e. initials of 2+ consecutive months)
    Returns
    -------
    month_list: list
        List of integers corresponding to the month number of each month in season (1=Jan,...)
    """
    if len(season_abbr) < 2:
        raise KeyError('Use seasons longer than one month')
    rolling_months = ''.join([m[0] for m in calendar.month_abbr[1:]])*2
    if season_abbr in rolling_months:
        season_start_idx = rolling_months.find(season_abbr)
        season_end_idx = season_start_idx + len(season_abbr)
        month_list = [(m%12)+1 for m in range(season_start_idx, season_end_idx)]
    else:
      raise KeyError(f'{season_abbr} not a valid sequence of months')
    return month_list


def mask_to_months(dates, data, month_list=np.arange(12)+1):
    """
    Mask any data outside of a fixed list of months to NaN.
    Parameters
    ----------
    dates: numpy array of decimal dates
        Dates corresponding to the data time stamps
    data: numpy array
        Data to be masked. Time must be dimension 0
    month_list (kwarg): list or numpy array
        List of permitted months, e.g. if month_list = [3, 4, 5]
        then only data in MAM will remain valid after masking.
        Default is for all months to be included.
    Returns
    -------
    masked_data: numpy array
        Data masked to selecte months
    """
    all_months = np.array([num2date(d, units='days since 1970-01-01', calendar='gregorian').month for d in dates])
    mask = ~np.isin(all_months, np.array(month_list))
    masked_data = np.copy(data)
    masked_data[mask] = np.nan
    return masked_data


def make_data_array(data_variable,
                    lon_west=-180, lon_east=180, lat_south=-30, lat_north=30,
                    min_year=2000, max_year=2018, month_list=np.arange(12)+1, percent_readings_required=30.):
    """
    Load data for a variable into memory and get time/lon/lat coordinates.
    Uses read_data_all_years() from read_data_iris module.
    See read_data_iris documentation for list of supported variables.
    Parameters
    ----------
    data_variable: utils.dataset.Dataset object
        Helper object for the dataset to be loaded.
    lon_west (kwarg): int or float
        Longitude of western boundary of area to load (in degrees east). Default -180.
    lon_east (kwarg): int or float
        Longitude of eastern boundary of area to load (in degrees east). Default 180.
    lat_south (kwarg): int or float
        Latitude of southern boundary of area to load (in degrees north). Default -30.
    lat_north (kwarg): int or float
        Latitude of northern boundary of area to load (in degrees north). Default 30.
    min_year (kwarg): int
        First year for which to load data. Default 2000.
    max_year (kwarg): int
        First year for which to load data. Default 2018.
    month_list (kwarg): list or numpy array (1D) of ints.
        Months to include in data, e.g. [3, 4, 5] for MAM. Default all months.
    percent_readings_required (kwarg): float
        Pixels with fewer than this percentage of valid observations between [min_year, max_year]
        will have all data masked. Default 30.
    Returns
    -------
    dates: numpy array (1D, float)
        Dates corresponding to observations. Units are days since 1970-01-01.
    data_array_mask_season: numpy array (3D, float)
        Time/lon/lat array of data, cropped to specific years and area from kwargs
    lats: numpy array (1D, float)
        Latitudes of pixel centres (degrees north)
    lons: numpy array (1D, float)
        Longitudes of pixel centres (degrees east)
    """

    data = read_data_all_years(data_variable,
                               min_year=min_year, max_year=max_year,
                               lon_west=lon_west, lon_east=lon_east, lat_south=lat_south, lat_north=lat_north)
    dates = data.coord('time').points
    lons = data.coord('longitude').points
    lats = data.coord('latitude').points
    data_array = ma.filled(data.data, np.nan)
    data_array_mask_season = mask_to_months(dates, data_array, month_list=month_list)
    possible_days = (~np.isnan(mask_to_months(dates, np.ones_like(data_array[:, 0, 0]), month_list=month_list))).astype(int).sum()
    number_readings = (~np.isnan(data_array_mask_season)).sum(axis=0)
    pc_readings = 100.*number_readings/possible_days

    # Calculate whether there are enough data in the time series to meet an
    # requested minimum percentage.  The requested percentage is adjusted
    # (reduced) to account for variables that are only available on a longer
    # time step than the time coordinate of the data array.  For pixels that
    # don't meet the required percentage, the whole time series is set to
    # missing data.
    percent_readings_required /= data_variable.data_period
    insufficient_readings = pc_readings < percent_readings_required

    data_array_mask_season[:, insufficient_readings] = np.nan

    return dates, data_array_mask_season, lats, lons


def make_data_arrays(reference_variable, response_variable,
                     lon_west=-180, lon_east=180, lat_south=-30, lat_north=30, 
                     min_year=2002, max_year=2016, return_coords=False,
                     monthly_anomalies=False, flip_reference_sign=False, flip_response_sign=False,
                     month_list=np.arange(12)+1, percent_readings_required=30.):
    """
    Load data for reference and response variables into memory and get time/lon/lat coordinates.
    Uses read_data_all_years() from read_data_iris module.
    See read_data_iris documentation for list of supported variables.
    Includes sign flipping/month masking/date conversion and padding/masking pixels with too few observations,
    so that output from this function can be fed straight to the cross-spectral analysis.
    Parameters
    ----------
    data_variable: utils.dataset.Dataset object
        Helper object for the dataset to be loaded.
    lon_west (kwarg): int or float
        Longitude of western boundary of area to load (in degrees east). Default -180.
    lon_east (kwarg): int or float
        Longitude of eastern boundary of area to load (in degrees east). Default 180.
    lat_south (kwarg): int or float
        Latitude of southern boundary of area to load (in degrees north). Default -30.
    lat_north (kwarg): int or float
        Latitude of northern boundary of area to load (in degrees north). Default 30.
    min_year (kwarg): int
        First year for which to load data. Default 2000.
    max_year (kwarg): int
        First year for which to load data. Default 2018.
    return_coords (kwarg): bool
        Return arrays of latitude and longitude in addition to dates and data. Default False.
        Useful for appending results to a single save file.
    monthly_anomalies (kwarg): bool
        After loading data, transform to anomalies from monthly climatology. Default False.
        Applies to both reference and response variables.
    flip_reference_sign (kwarg): bool
        Multiply reference variable data by -1. Default False.
        Useful if working with two variables that are negatively correlated.
    flip_response_sign (kwarg): bool
        Multiply reference variable data by -1. Default False.
        Useful if working with two variables that are negatively correlated.
    month_list (kwarg): list or numpy array (1D) of ints.
        Months to include in data, e.g. [3, 4, 5] for MAM. Default all months.
    percent_readings_required (kwarg): float
        Pixels with fewer than this percentage of valid observations between [min_year, max_year]
        will have all data masked. Default 30.
    Returns
    -------
    decimal_dates: numpy array (1D, float)
        Dates corresponding to observations, expressed as decimal dates (e.g. 1 Jan 2000 = 2000.0).
        Padded to contain dates for all observations (both reference and response variables).
    padded_reference: numpy array (3D, float)
        Time/lon/lat array of data for reference variable, cropped to specific years and area from kwargs
        and padded to match up with correct observation dates.
    padded_response: numpy array (3D, float)
        Time/lon/lat array of data for response variable, cropped to specific years and area from kwargs
        and padded to match up with correct observation dates.
    --- The following are returned ONLY IF return_coords is set to True: ---
    lats: numpy array (1D, float)
        Latitudes of pixel centres (degrees north)
    lons: numpy array (1D, float)
        Longitudes of pixel centres (degrees east)
    """
    reference_dates, reference_array, lats, lons = make_data_array(reference_variable,
                                                                   lon_west=lon_west, lon_east=lon_east, 
                                                                   lat_south=lat_south, lat_north=lat_north,
                                                                   min_year=min_year, max_year=max_year, month_list=month_list,
                                                                   percent_readings_required=percent_readings_required)
    response_dates, response_array, lats, lons = make_data_array(response_variable,
                                                                 lon_west=lon_west, lon_east=lon_east, 
                                                                 lat_south=lat_south, lat_north=lat_north,
                                                                 min_year=min_year, max_year=max_year, month_list=month_list,
                                                                 percent_readings_required=percent_readings_required)
    common_dates, padded_reference, padded_response = pad_match_time(reference_dates, response_dates,
                                                                     reference_array, response_array)
    decimal_dates = np.array([days_since_1970_to_decimal_year(date) for date in common_dates])

    land_fraction = read_land_sea_mask(lon_west=lon_west, lon_east=lon_east, lat_south=lat_south, lat_north=lat_north)
    sea = (land_fraction == 0.)
    padded_reference[:, sea] = np.nan
    padded_response[:, sea] = np.nan

    if monthly_anomalies:
        months = [decimal_year_to_datetime(d).month for d in decimal_dates]
        reference_monthly_means = []
        response_monthly_means = []
        reference_anomaly_time_series = []
        response_anomaly_time_series = []
        for m in np.arange(12)+1:
            month_idcs = np.where(np.array(months)==m)[0]
            references_month = padded_reference[month_idcs]
            reference_monthly_means.append(ma.mean(ma.masked_invalid(references_month)))
            responses_month = padded_response[month_idcs]
            response_monthly_means.append(ma.mean(ma.masked_invalid(responses_month)))
        for i in range(padded_reference.size):
            month = months[i]
            reference_mean_month = reference_monthly_means[month-1]
            response_mean_month = response_monthly_means[month-1]
            reference_anomaly_time_series.append(padded_reference[i] - reference_mean_month)
            response_anomaly_time_series.append(padded_response[i] - response_mean_month)
        padded_reference = np.array(reference_anomaly_time_series)
        padded_response = np.array(response_anomaly_time_series)

    if flip_reference_sign:
        padded_reference *= -1.
    if flip_response_sign:
        padded_response *= -1.
    
    if return_coords:
        return decimal_dates, padded_reference, padded_response, lats, lons
    else:
        return decimal_dates, padded_reference, padded_response


def create_input_file(reference_variable, response_variable, dates, reference_data, response_data,
                      save_filename):
    """
    Create a netCDF file with the correct time stamps/columns/formatting to be read in 
    for cross-spectral analysis by csagan.
    Parameters
    ----------
    reference_variable: utils.dataset.Dataset object
        Helper object for the reference dataset to be loaded.
    response_variable: utils.dataset.Dataset object
        Helper object for the response dataset to be loaded.
    dates: list or numpy array
        Time stamps of data as decimal dates
    reference_data: numpy array
        Data for reference variable
    response_data: numpy array
        Data for response variable
    save_filename: str
        Path to file for saving data (i.e. path to where CSA input file should be created)
    Returns
    -------
    None
    """
    # Only save time steps where both reference and response variable have valid data (required by csagan)
    valid_response_idcs = ~np.isnan(response_data)
    valid_reference_idcs = ~np.isnan(reference_data)
    valid_idcs = np.logical_and(valid_response_idcs, valid_reference_idcs)
    valid_dates = dates[valid_idcs]
    valid_references = reference_data[valid_idcs].data
    valid_responses = response_data[valid_idcs].data
    
    f = Dataset(save_filename, 'w', format='NETCDF4')
    f.description = f'{reference_variable.name} and {response_variable.name} daily time series'
    today = datetime.today()
    f.history = "Created " + today.strftime("%d/%m/%y")
    f.createDimension('time', None)
    days = f.createVariable('Time', 'f8', 'time')
    days[:] = valid_dates
    days.units = 'years'
    data_response = f.createVariable(response_variable.varname, 'f4', ('time'))
    data_response.units = response_variable.units
    data_reference = f.createVariable(reference_variable.varname, 'f4', ('time'))
    data_reference.units = reference_variable.units
    data_response[:] = valid_responses
    data_reference[:] = valid_references
    f.close()


def run_csagan(exe_filename, data_filename, netcdf, time_variable_name, 
               time_format, time_double, obs_variable_name, model_variable_name,
               frequency_units, ray_freq, model_units, time_earliest, pre_whiten, correct_bias):
    """
    Wrapper for running csagan.f. Includes all input options so that program will not require
    user interaction at runtime (essential for running over thousands of pixels).
    Not tested with any combination of input options other than the ones currently set in default_run.
    Parameters
    ----------
    exe_filename: str
        Path to the compiled csagan executable file
    data_filename: str
        Path to file containing input data (in format created by create_input_file function)
    netcdf: bool
        Is input in netCDF format? (False means ASCII - not tested)
    time_variable_name: str
        Name of variable in input file corresponding to time.
        Should be 'Time' for files generated using create_input_file.
    time_format: str
        Format of time data. Valid options are:
        'continuous': Date expressed as decimal, e.g. 00:00 1 Jan 2000 is 2000.000
        'integer seconds relative': Integer seconds relative to the start year
        'fixed time step': Integer fixed-length time step
        Only 'continuous' has been tested. Data generated using create_input_file
        has continuous format for time.
    time_double: bool
        Is time data stored as double precision?
    obs_variable_name: str
        Name of reference variable
    model_variable_name: str
        Name of response variable
    frequency_units: str
        Required frequency units to output given the units of the time stamps.
        e.g. year_day means time stamps are in years and frequency should be in cycles per day.
        See documentation in csagan.f for all options.
    ray_freq: bool
        Whether to change Rayleigh frequency from default (not set up to work for True yet)
    model_units: str
        Units of response data
    time_earliest: bool
        Is the first timestep in the stored data the earliest (True) or the latest (False)?
    pre_whiten: bool
        Pre-whiten data before performing cross-spectral analysis
    correct_bias: bool
        Correct bias in the power spectra using Monte Carlo simulations?
    Returns
    -------
    None
    """                       
    # Transform all options into the format required by csagan's user input prompts.
    netcdf = str(netcdf + 1) #convert from False/True to 1/2
    time_variable_codes = {'continuous': '1', 'integer seconds relative': '2', 'fixed time step': '3'}
    time_format = time_variable_codes[time_format]
    frequency_unit_codes = {'year_year': '1', 'day_day': '2', 'year_day': '3', 'hour_hour': '4',
                            'minute_minute': '5', 'second_second': 6}
    frequency_units = frequency_unit_codes[frequency_units]
    change_ray_freq = 'Y' if ray_freq else 'N'
    time_double = str(time_double + 1)
    correct_bias = 'Y' if correct_bias else 'N'
    time_earliest = 'E' if time_earliest else 'L'
    pre_whiten = str(int(pre_whiten)) # string 0 or 1

    # Get process ID from input filename - needs to be fed into csagan args so that output file
    # will have the same process ID in filename.
    process_id = data_filename.split('-')[-1].split('.')[0]
    # Put all the arguments to be fed to csagan together
    csagan_args = os.linesep.join([netcdf, process_id, data_filename, time_variable_name, time_format,
                                   time_double, obs_variable_name, model_variable_name, frequency_units,
                                   model_units, time_earliest, change_ray_freq, correct_bias])
    # Run csagan with set arguments to automatically answer when user input prompted.
    # Note set not to print or save stdout/stderr, remove these kwargs if output needed for debugging.
    csagan = run(exe_filename, text=True, input=csagan_args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                   
                   
def default_run(exe_filename, reference_variable, response_variable, data_filename):
    """
    Run cross-spectral analysis on a pre-created input netCDF file, with the csagan options
    needed for the analysis in Harris et al. 2023.
    Parameters
    ----------
    exe_filename: str
        Path to compiled csagan executable
    reference_variable: str
        Name of reference variable
    response_variable: str
        Name of response variable
    data_filename: str
        Path to input data file for csagan
    Returns
    -------
    spectra_results: dict
        Dictionary of arrays of results from cross-spectral analysis
    """
    # See documentation for run_csagan() function for more detail on csagan options
    netcdf = True
    time_variable_name = 'Time'
    time_format = 'continuous'
    time_double = True
    obs_variable_name = reference_variable
    model_variable_name = response_variable
    frequency_units = 'year_day'
    ray_freq = False
    model_units = 'unknown'
    time_earliest = True
    pre_whiten = False
    correct_bias = False
    run_csagan(exe_filename, data_filename, netcdf, time_variable_name, 
               time_format, time_double, obs_variable_name, model_variable_name,
               frequency_units, ray_freq, model_units, time_earliest, pre_whiten, correct_bias)
    # get process ID number - the ID in input filename will be automatically used in csagan output file
    process_id = int(data_filename.split('-')[-1].split('.')[0])    
    spectra_results = read_csagan_output(process_id)
    return spectra_results
                   
    
def delete_csagan_output(process_id, directory='.'):
    """
    Delete csagan output files by process ID number.
    Parameters
    ----------
    process_id: int
        ID of process for which to delete output
    directory (kwarg, default='.'): str
        Directory in which csagan results are output
    Returns
    -------
    None
    """
    os.remove(f'{directory}/csaout-{process_id}.nc')
    os.remove(f'{directory}/csaout-phase95-{process_id}.nc')


def delete_csagan_input(process_id, reference_variable, response_variable, directory='.'):
    """
    Delete csagan input files by process ID number.
    Parameters
    ----------
    process_id: int
        ID of process for which to delete output
    reference_variable: str
        Name of reference variable
    response_variable: str
        Name of response variable
    directory (kwarg, default='.'): str
        Directory in which csagan results are output
    Returns
    -------
    None
    """
    os.remove(f'{directory}/{reference_variable}_{response_variable}_input-{process_id}.nc')
    

def read_csagan_output(process_id, directory='.'):
    """
    Read results file generated by csagan.
    Parameters
    ----------
    process_id: int
        ID of process for which to read output
    directory (kwarg, default='.'): str
        Directory in which csagan results are output
    Returns
    -------
    spectra_results: dict
        Dictionary of arrays of results of cross-spectral analysis.
        Contains resolution bandwith, period, log power for reference and response variables,
        coherency, phase difference and its 95% confidence interval bounds, amplitude ratio
        and its 95% confidence interval bounds
    """
    spectra_filename = f'{directory}/csaout-{process_id}.nc'
    spectra_results = {}
    with Dataset(spectra_filename, 'r') as spectra_data:
        spectra_results['resolution_bandwidth'] = spectra_data.getncattr('Resolution_bandwidth')
        spectra_results['period'] = spectra_data.variables['period'][:]
        spectra_results['log_power_obs'] = spectra_data.variables['logpobs'][:]
        spectra_results['log_power_model'] = spectra_data.variables['logpmod'][:]
        spectra_results['coherency'] = spectra_data.variables['coherency'][:]
        spectra_results['phase'] = spectra_data.variables['phase'][:]
        spectra_results['phase_upper95'] = spectra_data.variables['ph95upr'][:]
        spectra_results['phase_lower95'] = spectra_data.variables['ph95lor'][:]
        spectra_results['amplitude'] = spectra_data.variables['amratio'][:]
        spectra_results['amplitude_upper95'] = spectra_data.variables['ar95upr'][:]
        spectra_results['amplitude_lower95'] = spectra_data.variables['ar95lor'][:]
    return spectra_results
    

def reference_response_spectra(exe_filename, work_directory, process_id,
                               reference_variable, response_variable, dates,
                               reference_data, response_data):
    """
    Perform cross-spectral analysis on a pixel, given data for two variables
    Parameters
    ----------
    exe_filename: str
        Path to compiled csagan executable
    work_directory: str
        Path to directory for work reference spectra.
    process_id: int
        Process ID for pixel computation
    reference_variable: utils.dataset.Dataset object
        Name of reference variable
    response_variable: utils.dataset.Dataset object
        Name of response variable
    dates: numpy array (1D, float)
        Decimal dates - same dates must be applicable to both reference and response data
    reference_data: numpy array (1D, float)
        Pixel data for reference variable
    response_data: numpy array (1D, float)
        Pixel data for response variable
    Returns
    -------
    spectra_results: dict
        Dictionary of arrays of results of cross-spectral analysis.
        Contains resolution bandwith, period, log power for reference and response variables,
        coherency, phase difference and its 95% confidence interval bounds, amplitude ratio
        and its 95% confidence interval bounds
    """
    data_filename = f'{work_directory}/{reference_variable.name}_{response_variable.name}_input-{process_id}.nc'
    try:
        create_input_file(reference_variable, response_variable, dates, reference_data, response_data,
                          data_filename)
        spectra_results = default_run(exe_filename, reference_variable.varname, response_variable.varname, data_filename)
        delete_csagan_output(process_id, directory='.')
        delete_csagan_input(process_id, reference_variable.name, response_variable.name, directory=work_directory)
    except Exception as e:
        print(f'***NO SPECTRA CREATED FOR PIXEL {process_id}***: {e}')
        spectra_results = {}
    return spectra_results


def csa_from_indices(coords):
    """
    Perform cross-spectral analysis for a single pixel.
    Written to work with multiprocessing using init_worker() function,
    relies on shared data having been created using the make_shared_array().
    Parameters
    ----------
    coords: tuple
        Coordinates of pixel in format (latitude_index, longitude_index)
    Returns
    -------
    spectra: dict
    Dictionary of cross-spectral analysis results (frequency, coherency, amplitude, phase difference) for pixel
    """
    csagan_exe = init_dict['csagan_exe']
    work_dir = init_dict['work_dir']
    response_variable = init_dict['response_variable']
    reference_variable = init_dict['reference_variable']
    decimal_dates = np.frombuffer(init_dict['dates']).reshape(init_dict['dates_shape'])
    reference_array = np.frombuffer(init_dict['reference']).reshape(init_dict['reference_shape'])
    response_array = np.frombuffer(init_dict['response']).reshape(init_dict['response_shape'])
    lat_idx, lon_idx = coords
    readings_reference = (~np.isnan(reference_array[:, lat_idx, lon_idx])).astype(int).sum()
    readings_response = (~np.isnan(response_array[:, lat_idx, lon_idx])).astype(int).sum()
    # Feed data to analysis function only if both reference and response variables have at least one reading
    # (e.g. skip ocean pixels for rainfall-vegetation analysis)
    sufficient_readings = (readings_reference > 0.) and (readings_response > 0.)
    if sufficient_readings:
        px_ids = np.frombuffer(init_dict['px_id']).reshape(init_dict['px_id_shape'])
        process_id = int(px_ids[lat_idx, lon_idx])
        spectra = reference_response_spectra(csagan_exe, work_dir, process_id,
                                             reference_variable, response_variable,
                                             decimal_dates, reference_array[:, lat_idx, lon_idx],
                                             response_array[:, lat_idx, lon_idx])
    else:
        spectra = {}
    return spectra


def make_shared_array(data_array, dtype=np.float64):
    """
    Create shared array of data for multiprocessing.
    NOTE: not tested on anything other than float64 data.
    Parameters
    ----------
    data_array: numpy array
        Array of data that will need to be accessed by workers from multiprocessing pool
    dtype (kwarg, default=np.float64): dtype
        Datatype in the numpy array
    Returns
    -------
    data_shared: numpy array
    data_array.shape: shape of the data array
    """
    data_shared = RawArray('d', data_array.size)
    data_shared_np = np.frombuffer(data_shared, dtype=dtype).reshape(data_array.shape)
    np.copyto(data_shared_np, data_array)
    return data_shared, data_array.shape


def init_worker(csagan_exe, work_dir, reference_variable, response_variable,
                decimal_dates, dates_shape, reference_array, reference_shape,
                response_array, response_shape, px_id_array, px_id_shape):
    """
    Helper function for the multiprocessing. Makes all the relevant data accessible by all the processes.
    """
    init_dict['csagan_exe'] = csagan_exe
    init_dict['work_dir'] = work_dir

    init_dict['reference_variable'] = reference_variable
    init_dict['response_variable'] = response_variable
    init_dict['dates'] = decimal_dates
    init_dict['dates_shape'] = dates_shape
    init_dict['reference'] = reference_array
    init_dict['reference_shape'] = reference_shape
    init_dict['response'] = response_array
    init_dict['response_shape'] = response_shape
    init_dict['px_id'] = px_id_array
    init_dict['px_id_shape'] = px_id_shape


def write_to_dataset(filename, results, results_lats, results_lons):
    """
    Save results of cross-spectral analysis to file.
    If the file already exists, it will be loaded and the new results written in.
    This allows analysis to be done in tiles without creating lots of files.
    Assumes files saved with names according to latitude bands:
    southern = 60S-25S
    tropics = 35S-35N
    northern = 25N-65S
    polar = 55N-80N
    Parameters
    ----------
    filename: str
        Path to file where cross-spectral results are to be stored
    results: numpy array (2D)
        Lat-lon grid of results of cross-spectral analysis produced by reference_response_spectra()
    results_lats: numpy array (1D)
        Array of latitude points for area of results
    results_lons: numpy array (1D)
        Array of longitude points for area of results
    Returns
    -------
    None
    """
    if 'tropics' in filename:
        lat_south = -35
        lat_north = 35
    elif 'northern' in filename:
        lat_south = 25
        lat_north = 65
    elif 'southern' in filename:
        lat_south = -60
        lat_north = -25
    elif 'polar' in filename:
        lat_south = 55
        lat_north = 80
    else:
        raise KeyError('Save filename not recognised as belonging to a defined latitude band')

    region_lats = np.arange(lat_south, lat_north, 0.25) + 0.5*0.25
    region_lons = np.arange(-180, 180, 0.25) + 0.5*0.25

    start_lat = np.argmin(np.abs(region_lats-results_lats.min()))
    end_lat = np.argmin(np.abs(region_lats-results_lats.max()))
    start_lon = np.argmin(np.abs(region_lons-results_lons.min()))
    end_lon = np.argmin(np.abs(region_lons-results_lons.max()))
 
    if Path(filename).is_file():
        with open(filename, 'rb') as fin:
            region_array = pickle.load(fin)
    else:
        region_array = np.empty((region_lats.size, region_lons.size), dtype=object)
        for i in range(region_lats.size):
            for j in range(region_lons.size):
                region_array[i, j] = {}

    for i, lat_idx in enumerate(range(start_lat, end_lat+1)):
        for j, lon_idx in enumerate(range(start_lon, end_lon+1)):
            region_array[lat_idx, lon_idx] = results[i, j]

    with open(filename, 'wb') as fout:
        pickle.dump(region_array, fout)

    return


def csa_multiprocess_tile(csagan_exe, nproc, work_dir, output_dir,
                          reference_variable, response_variable,
                          min_year, max_year, season, region_name,
                          lon_west, lon_east, lat_south, lat_north):

    months = season_from_abbr(season)

    # Create arrays of time and data for both reference and response variables.
    # Don't analyse pixels with fewer than percent_readings_required % of valid obs over all timesteps
    # Use flip_response_sign if a positive change in reference_variable leads to a negative change in response_variable (otherwise output lags won't make sense)
    decimal_dates, reference_array, response_array, lats, lons = make_data_arrays(reference_variable, response_variable,
                                                                                  lon_west=lon_west, lon_east=lon_east,
                                                                                  lat_south=lat_south, lat_north=lat_north,
                                                                                  min_year=min_year, max_year=max_year,
                                                                                  return_coords=True, flip_response_sign=False,
                                                                                  month_list=months,
                                                                                  percent_readings_required=30.)

    # Set up ID labels for each pixel and shared arrays to be used by multiprocessing pool
    total_pixels = reference_array[0, :, :].size

    lat_idcs = np.arange(response_array.shape[1])
    lon_idcs = np.arange(response_array.shape[2])
    LAT, LON = np.meshgrid(lat_idcs, lon_idcs)
    coords = zip(LAT.ravel(), LON.ravel())
    px_ids = np.arange(lats.size * lons.size).astype(int).reshape(LAT.T.shape)

    dates_shared, dates_shape = make_shared_array(decimal_dates)
    reference_shared, reference_shape = make_shared_array(reference_array)
    response_shared, response_shape = make_shared_array(response_array)
    px_id_shared, px_id_shape = make_shared_array(px_ids)

    shared_data = (
        csagan_exe, work_dir,
        reference_variable, response_variable,
        dates_shared, dates_shape,
        reference_shared, reference_shape,
        response_shared, response_shape,
        px_id_shared, px_id_shape
    )

    # Perform cross-spectral analysis for each pixel in selected region and save results with pickle
    print(f'start pool: {total_pixels} pixels')
    start = time.time()
    with Pool(processes=nproc, initializer=init_worker, initargs=shared_data) as pool:
        csa_output = pool.map(csa_from_indices, coords, chunksize=1)
    results = np.reshape(csa_output, response_shape[1:], order='F')
    end = time.time()
    dud_pixels = (results=={}).sum(axis=1).sum(axis=0)
    print(f'completed {total_pixels} pixels in {end-start} seconds, {dud_pixels} pixels did not have enough data for computation')

    file_tmp = f'{region_name}_{reference_variable.name}_{response_variable.name}_spectra_{season}_mask_sw_best85.pkl'
    path_tmp = os.path.join(output_dir, file_tmp)

    print(f"Writing output to {path_tmp}")
    write_to_dataset(path_tmp, results, lats, lons)

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

    seasons = metadata["spectra"].get("seasons", None)
    tiles = metadata["spectra"].get("tiles", None)

    ###########################################################################
    # Set up normal variables from yaml input dicts.
    ###########################################################################
    csagan_exe = metadata["multiprocess"].get("executable", None)
    nproc = metadata["multiprocess"].get("nproc", None)

    work_dir = output_dirs["spectra_tmp"]
    output_dir = output_dirs["spectra"]

    reference_variable = get_dataset(datasets["reference_var"])
    response_variable = get_dataset(datasets["response_var"])

    min_year = datasets["year_beg"]
    max_year = datasets["year_end"]

    ###########################################################################
    # Check that the input variables have been interpreted correctly.
    ###########################################################################
    if reference_variable is None:
        sys.exit(f"ERROR: Unrecognised reference variable name: {args.reference_var}")
    else:
        print(f"INFO: Reference variable {reference_variable.name}")

    if response_variable is None:
        sys.exit(f"ERROR: Unrecognised response variable name: {args.response_var}")
    else:
        print(f"INFO: Response variable {response_variable.name}")

    if min_year > max_year:
        sys.exit(f"ERROR: Start year after end year: {min_year} > {max_year}")
    else:
        print(f"INFO: Year range {min_year} to {max_year}")

    print("INFO: Requested seasons:")
    for season in seasons:
        print(f"INFO:     {season}")

    print("INFO: Requested tiles:")
    for tile_name, tile_coords in tiles.items():
        lon_west, lon_east, lat_south, lat_north = tile_coords
        print(f"INFO:     {tile_name}, west: {lon_west} deg, east: {lon_east} deg, south: {lat_south} deg, north: {lat_north} deg")

    # Ensure the various output directories exist before doing any heavy
    # calculations.
    for dirname in (work_dir, output_dir):
        print(f"Creating directory {dirname}")
        os.makedirs(dirname, exist_ok=True)

    ###########################################################################
    # Run the cross-spectral analysis for each requested region (tile) and for
    # each requested season.  Regions are divided up into subtiles for
    # processing to reduce RAM use.
    ###########################################################################
    print("INFO: Starting cross-spectral analysis.")

    # Width of tile subtiles in longitude degrees.
    subtile_lons = 30.0

    for tile_name, tile_coords in tiles.items():
        tile_west, tile_east, tile_south, tile_north = tile_coords

        subtile_coords = [
            (lon, min(lon + subtile_lons, tile_east), tile_south, tile_north)
            for lon in np.arange(tile_west, tile_east, subtile_lons)
        ]

        for coords in subtile_coords:
            for season in seasons:
                print(f"INFO: Processing subtile {coords} of {tile_name} tile for {season}")
                csa_multiprocess_tile(csagan_exe, nproc, work_dir, output_dir,
                                      reference_variable, response_variable,
                                      min_year, max_year, season, tile_name,
                                      *coords)

    return


if __name__ == '__main__':
    main()
