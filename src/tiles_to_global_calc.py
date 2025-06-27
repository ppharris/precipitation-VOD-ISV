from itertools import product
import pickle
import numpy as np
import os

from read_csagan_saved_output import read_region_data
import utils.load as ul


def tile_global_from_saved_spectra(spectra_save_dir,
                                   reference_var, response_var,
                                   season, band_days_lower, band_days_upper):
    tropics_filename = f"{spectra_save_dir}/spectra_nooverlap_tropics_{reference_var}_{response_var}_{season}_sw_filter_best85_{int(band_days_lower)}-{int(band_days_upper)}.pkl"
    northern_filename = f"{spectra_save_dir}/spectra_nooverlap_northern_{reference_var}_{response_var}_{season}_sw_filter_best85_{int(band_days_lower)}-{int(band_days_upper)}.pkl"
    southern_filename = f"{spectra_save_dir}/spectra_nooverlap_southern_{reference_var}_{response_var}_{season}_sw_filter_best85_{int(band_days_lower)}-{int(band_days_upper)}.pkl"

    with open(tropics_filename, 'rb') as f:
        spectra_tropics = pickle.load(f)

    with open(northern_filename, 'rb') as f:
        spectra_northern = pickle.load(f)

    with open(southern_filename, 'rb') as f:
        spectra_southern = pickle.load(f)

    spectra_global = {}
    for key in spectra_tropics.keys():
        spectra_global[key] = np.empty((440, 1440))
        spectra_global[key][0:100] = spectra_southern[key][20:-20]
        spectra_global[key][100:340] = spectra_tropics[key][20:-20]
        spectra_global[key][340:] = spectra_northern[key][20:-40]
    return spectra_global


def tile_global_validity(spectra_save_dir,
                         reference_var, response_var,
                         season):
    tropics_filename = f"{spectra_save_dir}/tropics_{reference_var}_{response_var}_spectra_{season}_mask_sw_best85.pkl"
    northern_filename = f"{spectra_save_dir}/northern_{reference_var}_{response_var}_spectra_{season}_mask_sw_best85.pkl"
    southern_filename = f"{spectra_save_dir}/southern_{reference_var}_{response_var}_spectra_{season}_mask_sw_best85.pkl"
    _, _, spectra_tropics = read_region_data(tropics_filename, 'tropics', -180, 180, -30, 30)
    _, _, spectra_northern = read_region_data(northern_filename, 'northern', -180, 180, 30, 55)
    _, _, spectra_southern = read_region_data(southern_filename, 'southern', -180, 180, -55, -30)
    no_csa_global = np.zeros((440, 1440), dtype='bool')
    no_csa_global[0:100] = (spectra_southern == {})
    no_csa_global[100:340] = (spectra_tropics == {})
    no_csa_global[340:] = (spectra_northern == {})
    return no_csa_global


def save_lags_to_file(output_dirs, datasets, bands, seasons):

    lag_data_dir = output_dirs["lag_data"]
    spectra_save_dir = output_dirs["spectra"]
    spectra_filtered_save_dir = output_dirs["spectra_filtered"]

    reference_var = datasets["reference_var"]
    response_var = datasets["response_var"]

    for season, (lower, upper) in product(seasons, bands):
        print(season, lower, upper)
        lag_dict = tile_global_from_saved_spectra(spectra_filtered_save_dir,
                                                  reference_var, response_var,
                                                  season, lower, upper)
        lag = lag_dict['lag']
        period = lag_dict['period']
        lag_error = lag_dict['lag_error']
        coherency = lag_dict['coherency']

        no_csa = tile_global_validity(spectra_save_dir,
                                      reference_var, response_var,
                                      season)

        np.save(os.path.join(lag_data_dir, f'lag_{season}_{lower}-{upper}.npy'), lag)
        np.save(os.path.join(lag_data_dir, f'lag_error_{season}_{lower}-{upper}.npy'), lag_error)
        np.save(os.path.join(lag_data_dir, f'period_{season}_{lower}-{upper}.npy'), period)
        np.save(os.path.join(lag_data_dir, f'no_csa_{season}_{lower}-{upper}.npy'), no_csa)
        np.save(os.path.join(lag_data_dir, f'coherency_{season}_{lower}-{upper}.npy'), coherency)

    return


def main():

    ###########################################################################
    # Parse command line args and load input file.
    ###########################################################################
    parser = ul.get_arg_parser()
    args = parser.parse_args()

    metadata = ul.load_yaml(args)

    output_dirs = metadata.get("output_dirs", None)
    bands = [tuple(b) for b in metadata["filter"].get("bands", None)]
    seasons = metadata["spectra"].get("seasons", None)

    datasets = metadata["datasets"]

    ul.check_dirs(output_dirs,
                  input_names=("spectra", "spectra_filtered"),
                  output_names=("lag_data", ))

    ###########################################################################
    # Run the analysis.
    ###########################################################################
    save_lags_to_file(output_dirs, datasets, bands, seasons)


if __name__ == '__main__':
    main()
