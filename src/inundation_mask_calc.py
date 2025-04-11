import numpy as np
import numpy.ma as ma
import iris.coord_categorisation

from read_data_iris import read_data_all_years, get_date_constraint
import utils_load as ul
from utils_datasets import CCI_SM, VOD, VOD_SW


def save_number_seasonal_vod_obs(output_dirs, vod_no_sw_mask, vod_sw_mask, season):

    save_dir = output_dirs["number_obs"]

    vod_no_sw_mask_season = vod_no_sw_mask.extract(iris.Constraint(clim_season=season))
    vod_sw_mask_season = vod_sw_mask.extract(iris.Constraint(clim_season=season))
    total_obs_no_sw_mask = vod_no_sw_mask_season.collapsed('time', iris.analysis.COUNT,
                                                           function=lambda values: values > -99.)
    total_obs_sw_mask = vod_sw_mask_season.collapsed('time', iris.analysis.COUNT,
                                                     function=lambda values: values > -99.)
    total_possible_obs = vod_no_sw_mask_season.coord('time').points.size
    np.save(f'{save_dir}/total_possible_obs_{season}.npy', total_possible_obs)
    np.save(f'{save_dir}/total_obs_no_sw_mask_{season}.npy', ma.filled(total_obs_no_sw_mask.data, 0.))
    np.save(f'{save_dir}/total_obs_sw_mask_{season}.npy', ma.filled(total_obs_sw_mask.data, 0.))


def save_ssm_seasonal_obs_numbers(output_dirs, seasons, min_year=2000, max_year=2016, date_range=None):

    save_dir = output_dirs["number_obs"]

    if date_range is None:
        date_range = get_date_constraint(start_year=min_year, start_month=1,
                                         end_year=max_year, end_month=12)

    ssm = read_data_all_years(CCI_SM, lon_west=-180, lon_east=180,
                              lat_south=-55, lat_north=55, min_year=min_year, max_year=max_year)

    ssm = ssm.extract(date_range)
    iris.coord_categorisation.add_season(ssm, 'time', seasons=seasons, name='clim_season')

    for season in seasons:
        ssm_season = ssm.extract(iris.Constraint(clim_season=season))
        total_obs_ssm = ssm_season.collapsed('time', iris.analysis.COUNT,
                                             function=lambda values: values > -99.)
        np.save(f'{save_dir}/total_obs_ssm_{season}.npy', ma.filled(total_obs_ssm.data, 0.))


def save_all_seasonal_obs_numbers(output_dirs, seasons, min_year=2000, max_year=2016, date_range=None):

    if date_range is None:
        date_range = get_date_constraint(start_year=min_year, start_month=1,
                                         end_year=max_year, end_month=12)

    vod_no_sw_mask = read_data_all_years(VOD, min_year=min_year, max_year=max_year,
                                         lon_west=-180, lon_east=180,
                                         lat_south=-55, lat_north=55)
    vod_sw_mask = read_data_all_years(VOD_SW, min_year=min_year, max_year=max_year,
                                      lon_west=-180, lon_east=180,
                                      lat_south=-55, lat_north=55)

    vod_no_sw_mask = vod_no_sw_mask.extract(date_range)
    vod_sw_mask = vod_sw_mask.extract(date_range)
    iris.coord_categorisation.add_season(vod_no_sw_mask, 'time', seasons=seasons, name='clim_season')
    iris.coord_categorisation.add_season(vod_sw_mask, 'time', seasons=seasons, name='clim_season')

    for season in seasons:
        save_number_seasonal_vod_obs(output_dirs, vod_no_sw_mask, vod_sw_mask, season)

    save_ssm_seasonal_obs_numbers(output_dirs, seasons, min_year=min_year, max_year=max_year)


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
    min_year, max_year = 2001, 2018

    save_all_seasonal_obs_numbers(output_dirs, seasons,
                                  min_year=min_year, max_year=max_year)

    return


if __name__ == '__main__':
    main()
