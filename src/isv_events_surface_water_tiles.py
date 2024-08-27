import dask.array as da
import netCDF4 as nc
from tqdm import tqdm
import numpy as np
import numpy.ma as ma
import gc
import pickle
from scipy.signal import argrelextrema
import iris
import os

from read_data_iris import read_data_all_years, daily_anomalies_normalised, daily_anomalies, monthly_anomalies_normalised
from bandpass_filters import lanczos_lowpass_filter_missing_data
import utils_load as ul


def remove_aux_coords(cube):
    for coord in cube.aux_coords:
        cube.remove_coord(coord)
    return


def get_chunks(cube, coord, chunksize):
    """Return an iterator of slices dividing up a cube dim coord into chunks.

    These slices can be used to divide up a cube into sub-cubes along a dim
    coord for proccessing, e.g., [ntime, nlat, nlon] -> [ntime, 0:10, nlon],
    [ntime, 11:20, nlon], [ntime, 21:30, nlon], ...

    """
    k_dim = cube.coord_dims(coord)[0]
    n_dim = cube.shape[k_dim]
    return (slice(k, k+chunksize) for k in range(0, n_dim, chunksize))


def process_in_chunks(processor, cube, chunksize, **kwargs):
    """Apply a function to a cube by applying it to sub-cubes chunked by latitude.

    This can help break up a full-cube calculation that uses a lot of RAM into
    a series of smaller calculations that require less RAM.

    """
    coord = "latitude"
    chunks = get_chunks(cube, coord, chunksize)

    cubes = []
    for chunk in chunks:
        print(chunk)
        cube_tmp = processor(cube[:, chunk, :], **kwargs)
        remove_aux_coords(cube_tmp)
        cubes.append(cube_tmp.copy(data=cube_tmp.data.astype(np.float32)))

    cubeo = iris.cube.CubeList.concatenate_cube(cubes)

    return cubeo


def save_anomalies(output_dirs, hem):

    save_dir = output_dirs["data_isv"]

    if hem == 'north':
        lat_south = 0
        lat_north = 60
    elif hem == 'south':
        lat_south = -60
        lat_north = 0
    else:
        raise KeyError(f'Hemisphere must be either north or south, not {hem}')

    latitude_chunksize = 24

    kw_save = {
        "fill_value": -999999.0,
        "chunksizes": (1, 240, 1440),
    }

    ###########################################################################
    # VODCA vegetation optical depth
    ###########################################################################
    vod = read_data_all_years('VOD', band='X', min_year=2000, max_year=2018,
                              lon_west=-180, lon_east=180, lat_south=lat_south, lat_north=lat_north, mask_surface_water=True)

    vod_single = process_in_chunks(daily_anomalies_normalised, vod, latitude_chunksize, detrend=True)
    file_out = os.path.join(save_dir, f'daily_detrended_vod_norm_anom_singleprecision_{hem}.nc')
    iris.save(vod_single, file_out, **kw_save)

    del vod
    del vod_single
    gc.collect()

    ###########################################################################
    # IMERG precip
    ###########################################################################
    imerg = read_data_all_years('IMERG', regridded=True, min_year=2000, max_year=2018,
                                lon_west=-180, lon_east=180, lat_south=lat_south, lat_north=lat_north)

    imerg_single = process_in_chunks(daily_anomalies_normalised, imerg, latitude_chunksize, detrend=True)
    file_out = os.path.join(save_dir, f'daily_detrended_imerg_norm_anom_singleprecision_{hem}.nc')
    iris.save(imerg_single, file_out, **kw_save)

    del imerg_single
    gc.collect()

    imerg_single_no_norm = process_in_chunks(daily_anomalies, imerg, latitude_chunksize, detrend=True)
    file_out = os.path.join(save_dir, f'daily_detrended_imerg_anom_singleprecision_{hem}.nc')
    iris.save(imerg_single_no_norm, file_out, **kw_save)

    del imerg
    del imerg_single_no_norm
    gc.collect()

    ###########################################################################
    # SM soil moisture
    ###########################################################################
    sm = read_data_all_years('SM', min_year=2000, max_year=2018,
                             lon_west=-180, lon_east=180, lat_south=lat_south, lat_north=lat_north)

    sm_single = process_in_chunks(daily_anomalies_normalised, sm, latitude_chunksize, detrend=True)
    file_out = os.path.join(save_dir, f'daily_detrended_sm_norm_anom_singleprecision_{hem}.nc')
    iris.save(sm_single, file_out, **kw_save)

    del sm
    del sm_single
    gc.collect()


def save_ndvi_anomalies(output_dirs, hem):

    save_dir = output_dirs["data_isv"]

    if hem == 'north':
        lat_south = 0
        lat_north = 60
    elif hem == 'south':
        lat_south = -60
        lat_north = 0
    else:
        raise KeyError(f'Hemisphere must be either north or south, not {hem}')

    latitude_chunksize = 24

    kw_save = {
        "fill_value": -999999.0,
        "chunksizes": (1, 240, 1440),
    }

    aqua = read_data_all_years('NDVI', modis_sensor='aqua', min_year=2000, max_year=2018,
                               lon_west=-180, lon_east=180, lat_south=lat_south, lat_north=lat_north)

    aqua_single = process_in_chunks(monthly_anomalies_normalised, aqua, latitude_chunksize, detrend=True)
    file_out = os.path.join(save_dir, f'monthly_detrended_ndvi_aqua_norm_anom_singleprecision_{hem}.nc')
    iris.save(aqua_single, file_out, **kw_save)

    del aqua_single
    gc.collect()

    terra = read_data_all_years('NDVI', modis_sensor='terra', min_year=2000, max_year=2018,
                                lon_west=-180, lon_east=180, lat_south=lat_south, lat_north=lat_north)

    terra_single = process_in_chunks(monthly_anomalies_normalised, terra, latitude_chunksize, detrend=True)
    file_out = os.path.join(save_dir, f'monthly_detrended_ndvi_terra_norm_anom_singleprecision_{hem}.nc')
    iris.save(terra_single, file_out, **kw_save)

    del terra_single
    gc.collect()


def merge_ndvi_anomalies(output_dirs, hem):

    save_dir = output_dirs["data_isv"]

    kw_save = {
        "fill_value": -999999.0,
        "chunksizes": (1, 240, 1440),
    }

    dask_chunks = (100, 240, 1440)

    file_aqua = os.path.join(save_dir, f'monthly_detrended_ndvi_aqua_norm_anom_singleprecision_{hem}.nc')
    file_terra = os.path.join(save_dir, f'monthly_detrended_ndvi_terra_norm_anom_singleprecision_{hem}.nc')

    # Read data as a lazy netCDF Dataset rather than a lazy Iris Cube so that
    # we can use Dask Numpy operations to reduce the memory overhead.  Trying
    # to do the same thing via the cube.data results in the whole array being
    # loaded into memory.
    ncid_a = nc.Dataset(file_aqua, "r")
    aqua_nd = ncid_a.variables["NDVI"]
    aqua_data = da.from_array(aqua_nd, chunks=dask_chunks)

    ncid_t = nc.Dataset(file_terra, "r")
    terra_nd = ncid_t.variables["NDVI"]
    terra_data = da.from_array(terra_nd, chunks=dask_chunks)

    merged_data = da.where(da.ma.getmaskarray(terra_data), aqua_data, terra_data)

    # Load one of the input files as a cube because this is a convenient way to
    # establish the cube structure for the merged output data.
    aqua_anom = iris.load_cube(file_aqua)
    merged_cube = aqua_anom.copy(data=merged_data)

    file_out = os.path.join(save_dir, f'monthly_detrended_ndvi_merged_norm_anom_singleprecision_{hem}.nc')
    iris.save(merged_cube, file_out, **kw_save)


def get_dates_for_box(imerg_lowfreq, lat_idx, lon_idx):
    filtered_imerg_px = imerg_lowfreq[:, lat_idx, lon_idx]
    candidate_maxima = argrelextrema(filtered_imerg_px, np.greater)[0]
    m = np.nanmean(filtered_imerg_px)
    s = np.nanstd(filtered_imerg_px)
    sig_season_maxima = filtered_imerg_px[candidate_maxima] > m + s
    sig_event_idx = candidate_maxima[sig_season_maxima]
    return sig_event_idx


def grid_coords_from_id(imerg_lowfreq, id):
    ids = np.arange(imerg_lowfreq[0].size).reshape(imerg_lowfreq.shape[1], imerg_lowfreq.shape[2])
    coords = np.where(ids==id)
    return coords[0][0], coords[1][0]


def get_all_events(imerg_lowfreq):
    events = []
    box_ids = np.arange(imerg_lowfreq[0].size)
    for box_id in tqdm(box_ids, desc='finding dates of ISV maxima'):
        lat_idx, lon_idx = grid_coords_from_id(imerg_lowfreq, box_id)
        sig_event_idx = get_dates_for_box(imerg_lowfreq, lat_idx, lon_idx)
        for event in range(len(sig_event_idx)):
            events.append(((lat_idx, lon_idx), sig_event_idx[event]))
    return events


def save_events(output_dirs, hem):

    data_dir = output_dirs["data_isv"]

    imerg_anomaly = iris.load(os.path.join(data_dir, f'daily_detrended_imerg_anom_singleprecision_{hem}.nc'))[0]
    imerg_anom = ma.filled(imerg_anomaly.data.astype(np.float32), np.nan)
    imerg_pad = np.ones((6940-imerg_anom.shape[0], imerg_anom.shape[1], imerg_anom.shape[2]), dtype=np.float32) * np.nan
    imerg_anom = np.concatenate((imerg_pad, imerg_anom))

    imerg_grid_size = imerg_anom[0].shape

    imerg_lowfreq = np.empty_like(imerg_anom, dtype=np.float32)
    for i in tqdm(range(imerg_anom.shape[1]), desc='filtering IMERG to ISV'):
        for j in range(imerg_anom.shape[2]):
            imerg_lowfreq[:, i, j] = lanczos_lowpass_filter_missing_data(imerg_anom[:, i, j], 1./25., 
                                                                         window=121, min_slice_size=100)

    events = get_all_events(imerg_lowfreq)

    file_out = os.path.join(data_dir, f'imerg_isv_events_lowpass_1std_{hem}.pkl')
    with open(file_out, 'wb') as f:
        pickle.dump(events, f)


def main():

    ###########################################################################
    # Parse command line args and load input file.
    ###########################################################################
    parser = ul.get_arg_parser()

    args = parser.parse_args()
    metadata = ul.load_yaml(args)
    output_dirs = metadata.get("output_dirs", None)

    ###########################################################################
    # Run the analysis.
    ###########################################################################
    print('save standardised anomalies N Hemisphere')
    save_anomalies(output_dirs, 'north')
    save_ndvi_anomalies(output_dirs, 'north')
    merge_ndvi_anomalies(output_dirs, 'north')

    print('save standardised anomalies S Hemisphere')
    save_anomalies(output_dirs, 'south')
    save_ndvi_anomalies(output_dirs, 'south')
    merge_ndvi_anomalies(output_dirs, 'south')

    print('find ISV events N Hemisphere')
    save_events(output_dirs, 'north')

    print('find ISV events S Hemisphere')
    save_events(output_dirs, 'south')


if __name__ == '__main__':
    main()
