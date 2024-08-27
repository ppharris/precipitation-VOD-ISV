import numpy as np
import pickle
import iris
import os

from read_data_iris import crop_cube
import utils_load as ul


def land_cover_codes(land_cover):
    codes = {'baresparse': [60],
             'shrub': [20],
             'herb': [30],
             'shrubherb': [20, 30],
             'crop': [40],
             'openforest': [121, 122, 124, 126],
             'closedforest': [111, 112, 113, 114, 115, 116]}
    return codes[land_cover]


def land_cover_mask(land_cover, lat_south, lat_north):
    lc = iris.load('/prj/nceo/bethar/copernicus_landcover_2018.nc')[0]
    lc_tropics = crop_cube(lc, -180, 180, lat_south, lat_north)
    land_cover_mask = np.isin(lc_tropics.data, land_cover_codes(land_cover))
    return land_cover_mask


def events_for_land_cover(events, land_cover, lat_south, lat_north):
    lc_mask = land_cover_mask(land_cover, lat_south, lat_north)
    lats = np.arange(lat_south, lat_north, 0.25) + 0.5 * 0.25
    lc_events = []
    for event in events:
        if lc_mask[event[0][0], event[0][1]] and np.abs(lats[event[0][0]])<55.:
            lc_events.append(event)
    return lc_events


def time_series_around_date(data_grid, lat_idx, lon_idx, date_idx, days_range=60):
    box_whole_time_series = data_grid[:, lat_idx, lon_idx]
    end_buffer = np.ones(days_range)*np.nan
    data_pad = np.hstack((end_buffer, box_whole_time_series, end_buffer))
    time_series = data_pad[date_idx+days_range-days_range:date_idx+days_range+(days_range+1)]
    return time_series


def composite_events_all_valid(events, data_grid, imerg_anom, vod_anom, sm_anom, days_range=60, existing_composite=None, existing_n=None):
    days_around = np.arange(-days_range, days_range+1)
    if existing_composite is not None:
        composite = existing_composite
        n = existing_n
        start_idx = 0
    else:
        event = events[0]
        composite = time_series_around_date(data_grid, event[0][0], event[0][1], event[1], days_range=days_range)
        vod_series = time_series_around_date(vod_anom, event[0][0], event[0][1], event[1], days_range=days_range)
        sm_series = time_series_around_date(sm_anom, event[0][0], event[0][1], event[1], days_range=days_range)
        imerg_series = time_series_around_date(imerg_anom, event[0][0], event[0][1], event[1], days_range=days_range)
        composite[np.logical_or(np.logical_or(np.isnan(vod_series), np.isnan(sm_series)), np.isnan(imerg_series))] = np.nan
        n = (~np.isnan(composite)).astype(float)
        start_idx = 1
    for event in events[start_idx:]:
        event_series = time_series_around_date(data_grid, event[0][0], event[0][1], event[1], days_range=days_range)
        vod_series = time_series_around_date(vod_anom, event[0][0], event[0][1], event[1], days_range=days_range)
        sm_series = time_series_around_date(sm_anom, event[0][0], event[0][1], event[1], days_range=days_range)
        imerg_series = time_series_around_date(imerg_anom, event[0][0], event[0][1], event[1], days_range=days_range)
        event_series[np.logical_or(np.logical_or(np.isnan(vod_series), np.isnan(sm_series)), np.isnan(imerg_series))] = np.nan
        additional_valid_day = np.logical_and(~np.isnan(event_series), ~np.isnan(composite))
        first_valid_day = np.logical_and(~np.isnan(event_series), np.isnan(composite))
        valid_days = np.logical_or(additional_valid_day, first_valid_day)
        n[valid_days] += 1
        composite[additional_valid_day] = composite[additional_valid_day] + (event_series[additional_valid_day] - composite[additional_valid_day])/n[additional_valid_day]
        composite[first_valid_day] = event_series[first_valid_day]
    return days_around, composite, n


def composite_events(events, data_grid, days_range=60, existing_composite=None, existing_n=None):
    days_around = np.arange(-days_range, days_range+1)
    if existing_composite is not None:
        composite = existing_composite
        n = existing_n
        start_idx = 0
    else:
        event = events[0]
        composite = time_series_around_date(data_grid, event[0][0], event[0][1], event[1], days_range=days_range)
        n = (~np.isnan(composite)).astype(float)
        start_idx = 1
    for event in events[start_idx:]:
        event_series = time_series_around_date(data_grid, event[0][0], event[0][1], event[1], days_range=days_range)
        additional_valid_day = np.logical_and(~np.isnan(event_series), ~np.isnan(composite))
        first_valid_day = np.logical_and(~np.isnan(event_series), np.isnan(composite))
        valid_days = np.logical_or(additional_valid_day, first_valid_day)
        n[valid_days] += 1
        composite[additional_valid_day] = composite[additional_valid_day] + (event_series[additional_valid_day] - composite[additional_valid_day])/n[additional_valid_day]
        composite[first_valid_day] = event_series[first_valid_day]
    return days_around, composite, n


def hemi_composites(output_dirs, hem, land_covers, days_range=60):

    if hem == 'north':
        lat_south = 0
        lat_north = 60
    elif hem == 'south':
        lat_south = -60
        lat_north = 0
    else:
        raise KeyError(f'Hemisphere must be either north or south, not {hem}')

    data_isv_dir = output_dirs["data_isv"]

    file_events = os.path.join(data_isv_dir, f'imerg_isv_events_lowpass_1std_{hem}.pkl')
    with open(file_events, 'rb') as f:
        events = pickle.load(f)

    file_imerg = os.path.join(data_isv_dir, f'daily_detrended_imerg_norm_anom_singleprecision_{hem}.nc')
    imerg_anomaly = iris.load_cube(file_imerg)

    imerg_anom = imerg_anomaly.data.filled(np.nan)
    imerg_pad_shape = (6940-imerg_anom.shape[0], imerg_anom.shape[1], imerg_anom.shape[2])
    imerg_pad = np.full(imerg_pad_shape, np.nan, dtype=np.float32)
    imerg_anom = np.concatenate((imerg_pad, imerg_anom))

    file_vod = os.path.join(data_isv_dir, f'daily_detrended_vod_norm_anom_singleprecision_{hem}.nc')
    vod_anomaly = iris.load_cube(file_vod)
    vod_anom = vod_anomaly.data.filled(np.nan)

    file_sm = os.path.join(data_isv_dir, f'daily_detrended_sm_norm_anom_singleprecision_{hem}.nc')
    sm_anomaly = iris.load_cube(file_sm)
    sm_anom = sm_anomaly.data.filled(np.nan)

    for arr in [imerg_anom, vod_anom, sm_anom]:
        print(arr.shape)

    data_comp_dir = output_dirs["data_isv_comp"]
    
    for land_cover in land_covers:
        lc_events = events_for_land_cover(events, land_cover, lat_south, lat_north)
        days_around, imerg_composite, imerg_n = composite_events_all_valid(lc_events, imerg_anom, imerg_anom, vod_anom, sm_anom, days_range=days_range)
        _, vod_composite, vod_n = composite_events_all_valid(lc_events, vod_anom, imerg_anom, vod_anom, sm_anom, days_range=days_range)
        _, sm_composite, sm_n = composite_events_all_valid(lc_events, sm_anom, imerg_anom, vod_anom, sm_anom, days_range=days_range)

        if np.logical_and(np.allclose(imerg_n, vod_n), np.allclose(imerg_n, sm_n)):
            n = imerg_n
        else:
            raise ValueError('ns are not the same')

        file_out = os.path.join(data_comp_dir, f'imerg_composite_{hem}_55NS_{land_cover}.npy')
        np.save(file_out, imerg_composite)

        file_out = os.path.join(data_comp_dir, f'vod_composite_{hem}_55NS_{land_cover}.npy')
        np.save(file_out, vod_composite)

        file_out = os.path.join(data_comp_dir, f'sm_composite_{hem}_55NS_{land_cover}.npy')
        np.save(file_out, sm_composite)

        file_out = os.path.join(data_comp_dir, f'n_{hem}_55NS_{land_cover}.npy')
        np.save(file_out, n)


def full_composites(output_dirs, land_covers, days_range=60):
    # must have already run hemi_composites('south')

    data_isv_dir = output_dirs["data_isv"]

    file_events = os.path.join(data_isv_dir, 'imerg_isv_events_lowpass_1std_north.pkl')
    with open(file_events, 'rb') as f:
        events = pickle.load(f)

    file_imerg = os.path.join(data_isv_dir, 'daily_detrended_imerg_norm_anom_singleprecision_north.nc')
    imerg_anomaly = iris.load_cube(file_imerg)

    imerg_anom = imerg_anomaly.data.filled(np.nan)
    imerg_pad_shape = (6940-imerg_anom.shape[0], imerg_anom.shape[1], imerg_anom.shape[2])
    imerg_pad = np.full(imerg_pad_shape, np.nan, dtype=np.float32)
    imerg_anom = np.concatenate((imerg_pad, imerg_anom))

    file_vod = os.path.join(data_isv_dir, 'daily_detrended_vod_norm_anom_singleprecision_north.nc')
    vod_anomaly = iris.load_cube(file_vod)
    vod_anom = vod_anomaly.data.filled(np.nan)

    file_sm = os.path.join(data_isv_dir, 'daily_detrended_sm_norm_anom_singleprecision_north.nc')
    sm_anomaly = iris.load_cube(file_sm)
    sm_anom = sm_anomaly.data.filled(np.nan)

    data_comp_dir = output_dirs["data_isv_comp"]

    lat_south = 0
    lat_north = 60
    for land_cover in land_covers:
        lc_events = events_for_land_cover(events, land_cover, lat_south, lat_north)

        imerg_composite_south = np.load(os.path.join(data_comp_dir, f'imerg_composite_south_55NS_{land_cover}.npy'))
        vod_composite_south = np.load(os.path.join(data_comp_dir, f'vod_composite_south_55NS_{land_cover}.npy'))
        sm_composite_south = np.load(os.path.join(data_comp_dir, f'sm_composite_south_55NS_{land_cover}.npy'))
        n_south = np.load(os.path.join(data_comp_dir, f'n_south_55NS_{land_cover}.npy'))

        days_around, imerg_composite, imerg_n = composite_events_all_valid(lc_events, imerg_anom, imerg_anom, vod_anom, sm_anom, 
                                                                           days_range=days_range, existing_composite=imerg_composite_south, existing_n=n_south)
        _, vod_composite, vod_n = composite_events_all_valid(lc_events, vod_anom, imerg_anom, vod_anom, sm_anom, 
                                                             days_range=days_range, existing_composite=vod_composite_south, existing_n=n_south)
        _, sm_composite, sm_n = composite_events_all_valid(lc_events, sm_anom, imerg_anom, vod_anom, sm_anom, 
                                                           days_range=days_range, existing_composite=sm_composite_south, existing_n=n_south)

        if np.logical_and(np.allclose(imerg_n, vod_n), np.allclose(imerg_n, sm_n)):
            n = imerg_n
        else:
            raise ValueError('ns are not the same')

        file_out = os.path.join(data_comp_dir, f'imerg_composite_global_55NS_{land_cover}.npy')
        np.save(file_out, imerg_composite)

        file_out = os.path.join(data_comp_dir, f'vod_composite_global_55NS_{land_cover}.npy')
        np.save(file_out, vod_composite)

        file_out = os.path.join(data_comp_dir, f'sm_composite_global_55NS_{land_cover}.npy')
        np.save(file_out, sm_composite)

        file_out = os.path.join(data_comp_dir, f'n_global_55NS_{land_cover}.npy')
        np.save(file_out, n)


def ndvi_composites(output_dirs, land_covers, days_range=60):

    data_isv_dir = output_dirs["data_isv"]
    data_comp_dir = output_dirs["data_isv_comp"]

    ###########################################################################
    # S Hem first
    ###########################################################################
    lat_south = -60
    lat_north = 0

    file_events = os.path.join(data_isv_dir, f'imerg_isv_events_lowpass_1std_south.pkl')
    with open(file_events, 'rb') as f:
        events = pickle.load(f)

    file_ndvi = os.path.join(data_isv_dir, f'monthly_detrended_ndvi_merged_norm_anom_singleprecision_south.nc')
    ndvi_anomaly = iris.load_cube(file_ndvi)
    ndvi_anom = ndvi_anomaly.data.filled(np.nan)

    for land_cover in land_covers:
        lc_events = events_for_land_cover(events, land_cover, lat_south, lat_north)
        days_around, ndvi_composite, ndvi_n = composite_events(lc_events, ndvi_anom, days_range=days_range)

        file_out = os.path.join(data_comp_dir, f'ndvi_composite_south_55NS_{land_cover}.npy')
        np.save(file_out, ndvi_composite)

        file_out = os.path.join(data_comp_dir, f'n_south_55NS_{land_cover}.npy')
        np.save(file_out, ndvi_n)

    ###########################################################################
    # add N Hem
    ###########################################################################
    lat_south = 0
    lat_north = 60

    file_events = os.path.join(data_isv_dir, f'imerg_isv_events_lowpass_1std_north.pkl')
    with open(file_events, 'rb') as f:
        events = pickle.load(f)

    file_ndvi = os.path.join(data_isv_dir, f'monthly_detrended_ndvi_merged_norm_anom_singleprecision_north.nc')
    ndvi_anomaly = iris.load_cube(file_ndvi)
    ndvi_anom = ndvi_anomaly.data.filled(np.nan)

    for land_cover in land_covers:
        lc_events = events_for_land_cover(events, land_cover, lat_south, lat_north)

        ndvi_composite_south = np.load(os.path.join(data_comp_dir, f'ndvi_composite_south_55NS_{land_cover}.npy'))
        n_south = np.load(os.path.join(data_comp_dir, f'n_south_55NS_{land_cover}.npy'))

        days_around, ndvi_composite, ndvi_n = composite_events(lc_events, ndvi_anom,
                                                               days_range=days_range, existing_composite=ndvi_composite_south,
                                                               existing_n=n_south)

        file_out = os.path.join(data_comp_dir, f'ndvi_composite_global_55NS_{land_cover}.npy')
        np.save(file_out, ndvi_composite)

        file_out = os.path.join(data_comp_dir, f'n_global_55NS_{land_cover}.npy')
        np.save(file_out, ndvi_n)


def main():

    ###########################################################################
    # Parse command line args and load input file.
    ###########################################################################
    parser = ul.get_arg_parser()
    args = parser.parse_args()

    metadata = ul.load_yaml(args)

    output_dirs = metadata.get("output_dirs", None)
    land_covers = metadata["isv"].get("land_covers", None)
    days_range = metadata["isv"].get("days_range", None)

    ###########################################################################
    # Run the analysis.
    ###########################################################################
    hemi_composites(output_dirs, 'south', land_covers)
    full_composites(output_dirs, land_covers)
    ndvi_composites(output_dirs, land_covers, days_range=days_range)


if __name__ == '__main__':
    main()
