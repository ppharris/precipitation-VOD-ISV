import os
from tqdm import tqdm
import numpy as np
import numpy.ma as ma
import iris.coord_categorisation
from iris.time import PartialDateTime

from read_data_iris import read_data_all_years
import utils_load as ul


def save_number_seasonal_vod_obs(output_dirs, vod_no_sw_mask, vod_sw_mask, season):

    save_dir = output_dirs["number_obs"]

    vod_no_sw_mask_season = vod_no_sw_mask.extract(iris.Constraint(clim_season=season.lower()))
    vod_sw_mask_season = vod_sw_mask.extract(iris.Constraint(clim_season=season.lower()))
    total_obs_no_sw_mask = vod_no_sw_mask_season.collapsed('time', iris.analysis.COUNT,
                                                           function=lambda values: values > -99.)
    total_obs_sw_mask = vod_sw_mask_season.collapsed('time', iris.analysis.COUNT,
                                                     function=lambda values: values > -99.)
    total_possible_obs = vod_no_sw_mask_season.coord('time').points.size
    np.save(f'{save_dir}/total_possible_obs_{season}.npy', total_possible_obs)
    np.save(f'{save_dir}/total_obs_no_sw_mask_{season}.npy', ma.filled(total_obs_no_sw_mask.data, 0.))
    np.save(f'{save_dir}/total_obs_sw_mask_{season}.npy', ma.filled(total_obs_sw_mask.data, 0.))


def save_ssm_seasonal_obs_numbers(output_dirs, seasons):

    save_dir = output_dirs["number_obs"]

    ssm = read_data_all_years('SM', lon_west=-180, lon_east=180, 
                              lat_south=-55, lat_north=55, min_year=2000, max_year=2018)
    jun2000 = PartialDateTime(year=2000, month=6)
    dec2018 = PartialDateTime(year=2018, month=31)
    date_range = iris.Constraint(time=lambda cell: jun2000 <= cell.point <= dec2018)
    ssm = ssm.extract(date_range)
    iris.coord_categorisation.add_season(ssm, 'time', name='clim_season')

    for season in tqdm(seasons, desc='saving SSM seasonal obs numbers'):
        ssm_season = ssm.extract(iris.Constraint(clim_season=season.lower()))
        total_obs_ssm = ssm_season.collapsed('time', iris.analysis.COUNT,
                                             function=lambda values: values > -99.)
        np.save(f'{save_dir}/total_obs_ssm_{season}.npy', ma.filled(total_obs_ssm.data, 0.))


def save_all_seasonal_obs_numbers(output_dirs, seasons):
    vod_no_sw_mask = read_data_all_years('VOD', band='X', lon_west=-180, lon_east=180, 
                                         lat_south=-55, lat_north=55, min_year=2000, max_year=2018,
                                         mask_surface_water=False)
    vod_sw_mask = read_data_all_years('VOD', band='X', lon_west=-180, lon_east=180, 
                                      lat_south=-55, lat_north=55, min_year=2000, max_year=2018,
                                      mask_surface_water=True)
    jun2000 = PartialDateTime(year=2000, month=6)
    dec2018 = PartialDateTime(year=2018, month=31)
    date_range = iris.Constraint(time=lambda cell: jun2000 <= cell.point <= dec2018)
    vod_no_sw_mask = vod_no_sw_mask.extract(date_range)
    vod_sw_mask = vod_sw_mask.extract(date_range)
    iris.coord_categorisation.add_season(vod_no_sw_mask, 'time', name='clim_season')
    iris.coord_categorisation.add_season(vod_sw_mask, 'time', name='clim_season')
    for season in tqdm(seasons, desc='saving seasonal obs numbers'):
        save_number_seasonal_vod_obs(output_dirs, vod_no_sw_mask, vod_sw_mask, season)
    save_ssm_seasonal_obs_numbers(output_dirs, seasons)


def main():

    ###########################################################################
    # Parse command line args and load input file.
    ###########################################################################
    parser = ul.get_arg_parser()
    args = parser.parse_args()

    metadata = ul.load_yaml(args)

    output_dirs = metadata.get("output_dirs", None)
    seasons = metadata["lags"].get("seasons", None)

    ul.check_dirs(output_dirs,
                  output_names=("number_obs",)
    )

    ###########################################################################
    # Run the analysis.
    ###########################################################################
    save_all_seasonal_obs_numbers(output_dirs, seasons)

    return


if __name__ == '__main__':
    main()
