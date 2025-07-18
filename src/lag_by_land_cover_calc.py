import numpy as np
import os
import re
import iris
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from tiles_to_global_calc import tile_global_validity, tile_global_from_saved_spectra
from read_csagan_saved_output import read_region_data
from read_data_iris import crop_cube
import utils.load as ul
from utils.plot import binned_cmap, StripyPatch


lats_south = {'global': -55, 'southern': -60, 'tropics': -30, 'northern': 30, 'polar': 60}
lats_north = {'global': 55, 'southern': -30, 'tropics': 30, 'northern': 60, 'polar': 80}


def copernicus_land_cover(lon_west=-180, lon_east=180, lat_south=-60, lat_north=60):
    lc = iris.load('/prj/nceo/bethar/copernicus_landcover_2018.nc')[0]
    lc_region = crop_cube(lc, lon_west, lon_east, lat_south, lat_north)
    coded_land_cover_array = lc_region.data
    string_array = np.empty((coded_land_cover_array.shape[0], coded_land_cover_array.shape[1]), dtype='U22')
    code_dict = {
    0: 'Missing',
    20: 'Shrubland',
    30: 'Herbaceous vegetation',
    40: 'Cropland',
    50: 'Urban',
    60: 'Bare/sparse vegetation',
    70: 'Snow/ice',
    80: 'Inland water',
    90: 'Herbaceous wetland',
    100: 'Moss/lichen',
    111: 'Closed forest',
    112: 'Closed forest',
    113: 'Closed forest',
    114: 'Closed forest',
    115: 'Closed forest',
    116: 'Closed forest',
    121: 'Open forest',
    122: 'Open forest',
    123: 'Open forest',
    124: 'Open forest',
    125: 'Open forest',
    126: 'Open forest',
    200: 'Open water'
    }
    for i in range(coded_land_cover_array.shape[0]):
        for j in range(coded_land_cover_array.shape[1]):
            code = coded_land_cover_array[i, j]
            string_array[i, j] = code_dict[code]
    return string_array


def line_break_string(the_string, max_line_length):
    if len(the_string) <= max_line_length:
        return the_string
    else:
        spaces = re.finditer(' ', the_string)
        space_positions = np.array([space.start() for space in spaces])
        if np.any(space_positions < max_line_length):
            space_position = space_positions[space_positions < max_line_length][-1]
        else:
            space_position = space_positions[0]
        broken_string = the_string[0:space_position] + '\n' + the_string[space_position+1:]
        return broken_string


def load_num_obs(number_obs_dir, seasons):

    num_obs = {}
    for season in seasons:
        season_obs = {
            "num_total": np.load(os.path.join(number_obs_dir, f'total_possible_obs_{season}.npy')),
            "num_before": np.load(os.path.join(number_obs_dir, f'total_obs_no_sw_mask_{season}.npy')),
            "num_after": np.load(os.path.join(number_obs_dir, f'total_obs_sw_mask_{season}.npy')),
        }
        num_obs[season] = season_obs
    return num_obs


def all_season_lags(output_dirs, datasets, tile, band_days_lower, band_days_upper, seasons, nonzero_only=False):

    spectra_filt_dir = output_dirs["spectra_filtered"]

    reference_var = datasets["reference_var"]
    response_var = datasets["response_var"]

    mask_lag = {}
    for season in seasons:
        if tile != 'global':
            tile_filename = f"{spectra_save_dir}/spectra_nooverlap_{tile}_{reference_var}_{response_var}_{season}_sw_filter_best85_{int(band_days_lower)}-{int(band_days_upper)}.pkl"
            with open(tile_filename,'rb') as f:
                neighbourhood_average_spectra = pickle.load(f)
            # Subset array for some reason?
            for key in neighbourhood_average_spectra.keys():
                neighbourhood_average_spectra[key] = neighbourhood_average_spectra[key][20:-20]
        else:
            neighbourhood_average_spectra = tile_global_from_saved_spectra(spectra_filt_dir,
                                                                           reference_var, response_var,
                                                                           season, band_days_lower, band_days_upper)
        if nonzero_only:
            mask_lag_mean = neighbourhood_average_spectra['lag']
            mask_lag_error = neighbourhood_average_spectra['lag_error']
            lags_nonzero_only = np.copy(mask_lag_mean)
            lags_upper = lags_nonzero_only + mask_lag_error
            lags_lower = lags_nonzero_only - mask_lag_error
            confidence_interval_overlaps_zero = (np.sign(lags_lower)/np.sign(lags_upper) == -1)
            lags_nonzero_only[confidence_interval_overlaps_zero] = np.nan
            mask_lag[season] = lags_nonzero_only
        else:
            mask_lag[season] = neighbourhood_average_spectra['lag']
    # mask_lag is an ndarray.
    return mask_lag


def median_95ci_width(output_dirs, datasets, tile, band_days_lower, band_days_upper, seasons):

    spectra_save_dir = output_dirs["spectra_filtered"]

    reference_var = datasets["reference_var"]
    response_var = datasets["response_var"]

    all_lag_errors = {}
    for season in seasons:
        if tile != 'global':
            tile_filename = f"{spectra_save_dir}/spectra_nooverlap_{tile}_{reference_var}_{response_var}_{season}_sw_filter_best85_{int(band_days_lower)}-{int(band_days_upper)}.pkl"
            with open(tile_filename,'rb') as f:
                neighbourhood_average_spectra = pickle.load(f)
            for key in neighbourhood_average_spectra.keys():
                if tile == 'polar':
                    neighbourhood_average_spectra[key] = neighbourhood_average_spectra[key][20:-40]
                else:
                    neighbourhood_average_spectra[key] = neighbourhood_average_spectra[key][20:-20]
        else:
            neighbourhood_average_spectra = tile_global_from_saved_spectra(spectra_save_dir,
                                                                           reference_var, response_var,
                                                                           season, band_days_lower, band_days_upper)
        all_lag_errors[season] = neighbourhood_average_spectra['lag_error']
    lag_errors_stack = np.stack([v for v in all_lag_errors.values()])
    return np.nanpercentile(lag_errors_stack, 50)


def all_season_validity(output_dirs, datasets, tile, band_days_lower, band_days_upper, seasons):

    spectra_save_dir = output_dirs["spectra"]
    spectra_filt_dir = output_dirs["spectra_filtered"]

    reference_var = datasets["reference_var"]
    response_var = datasets["response_var"]

    validity = {}
    for season in seasons:
        if tile != 'global':
            tile_filename = f"{spectra_filt_dir}/spectra_nooverlap_tropics_{reference_var}_{response_var}_{season}_sw_filter_best85_{int(band_days_lower)}-{int(band_days_upper)}.pkl"
            with open(tile_filename,'rb') as f:
                neighbourhood_average_spectra = pickle.load(f)

            original_output_filename = f"{spectra_save_dir}/tropics_{reference_var}_{response_var}_spectra_{season}_mask_sw_best85.pkl"
            lats, lons, spectra = read_region_data(original_output_filename, tile, -180, 180, lats_south[tile], lats_north[tile])
            no_csa = (spectra == {})
            for key in neighbourhood_average_spectra.keys():
                if tile == 'polar':
                    neighbourhood_average_spectra[key] = neighbourhood_average_spectra[key][20:-40]
                else:
                    neighbourhood_average_spectra[key] = neighbourhood_average_spectra[key][20:-20]
        else:
            neighbourhood_average_spectra = tile_global_from_saved_spectra(spectra_filt_dir,
                                                                           reference_var, response_var,
                                                                           season, band_days_lower, band_days_upper)
            no_csa = tile_global_validity(spectra_save_dir,
                                          reference_var, response_var,
                                          season)
        coherencies = neighbourhood_average_spectra['coherency']
        validity[season] = (coherencies >= 0.7795).astype(int)
        validity[season][no_csa] = 2
    return validity


def hist_lc(mask_lag, lc_codes, land_cover_code, lag_bin_bounds, density=False):
    all_lags = []
    for season in mask_lag.keys():
        if land_cover_code == 'all':
            all_lc_codes = ['Bare/sparse vegetation','Herbaceous vegetation',
                            'Shrubland','Cropland','Open forest', 'Closed forest']
            season_lags = mask_lag[season][np.isin(lc_codes, all_lc_codes)]
        else:
            season_lags = mask_lag[season][lc_codes==land_cover_code]
        all_lags += season_lags.tolist()
    hist, _ = np.histogram(all_lags, bins=lag_bin_bounds, density=density)
    return hist


def subplots(output_dirs, lag_data, median_data, lag_bin_bounds,
             density=False, show_95ci=True, all_lc_line=False,
             filename_out=None, plot_type="png"):

    figures_dir = output_dirs["figures"]

    if len(lag_data) > 2:
        print("WARN: Only plotting phase for first two bands.")

    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(2, 4)
    ax1 = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(gs[1, 0:2])
    ax3 = fig.add_subplot(gs[1, 2:])
    plt.subplots_adjust(hspace=0.45, wspace=0.4)

    lc_codes = copernicus_land_cover(lat_south=-55, lat_north=55)

    lc_array = np.zeros_like(lc_codes, dtype=int)
    lc_array[lc_codes=='Bare/sparse vegetation'] = 1
    lc_array[lc_codes=='Herbaceous vegetation'] = 2
    lc_array[lc_codes=='Shrubland'] = 3
    lc_array[lc_codes=='Cropland'] = 4
    lc_array[lc_codes=='Open forest'] = 5
    lc_array[lc_codes=='Closed forest'] = 6

    lons = np.arange(-180, 180, 0.25) + 0.125
    lats = np.arange(-55, 55, 0.25) + 0.125
    lon_bounds = np.hstack((lons - 0.5*0.25, np.array([lons[-1]+0.5*0.25])))
    lat_bounds = np.hstack((lats - 0.5*0.25, np.array([lats[-1]+0.5*0.25])))
    levels = np.arange(1, 8) - 0.5
    lc_colors = {'Bare/sparse vegetation': '#ffd92f',
                 'Herbaceous vegetation': '#EA8294',
                 'Shrubland': '#AA4499',
                 'Cropland': '#88CCEE',
                 'Open forest': '#889933',
                 'Closed forest': '#117733'}
    color_list = list(lc_colors.values())
    lc_cmap, lc_norm = binned_cmap(levels, 'tab10', fix_colours=[(i, c) for i, c in enumerate(color_list)])
    lc_cmap.set_bad('w')
    lc_cmap.set_under('w')

    p = ax1.pcolormesh(lon_bounds, lat_bounds, lc_array, transform=ccrs.PlateCarree(), 
                       cmap=lc_cmap, norm=lc_norm, rasterized=True)
    fig_left = ax2.get_position().x0
    fig_right = ax3.get_position().x1
    cax = fig.add_axes([fig_left, ax1.get_position().y0-0.075, fig_right-fig_left, 0.03])
    cbar = fig.colorbar(p, orientation='horizontal', cax=cax, aspect=40, pad=0.12)
    cbar.set_ticks(levels[:-1] + 0.5)
    cbar.ax.set_xticklabels(['Bare/sparse \n vegetation', 'Herbaceous \n vegetation', 'Shrubland', 'Cropland', 'Open forest', 'Closed forest'])
    cbar.ax.tick_params(labelsize=14)
    ax1.set_extent((-180, 180, -55, 55), crs=ccrs.PlateCarree())
    ax1.coastlines(color='black', linewidth=1)
    ax1.set_xticks(np.arange(-90, 91, 90), crs=ccrs.PlateCarree())
    ax1.set_yticks(np.arange(-50, 51, 50), crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    ax1.tick_params(labelsize=14)
    ax1.tick_params(axis='x', pad=5)
    ax1.set_title("$\\bf{(a)}$" + ' Modal land cover class (Copernicus 2018)', fontsize=14)

    lc_codes = copernicus_land_cover(lat_south=-55, lat_north=55)

    bin_centres = lag_bin_bounds[:-1] + (lag_bin_bounds[1] - lag_bin_bounds[0])/2.

    for ax, letter, (band, lags) in zip([ax2, ax3], "bc", lag_data.items()):
        for land_cover_code in lc_colors.keys():
            hist = hist_lc(lags, lc_codes, land_cover_code, lag_bin_bounds, density=density)
            ax.plot(bin_centres, hist, '-o', color=lc_colors[land_cover_code], ms=2.5)

        if all_lc_line:
            hist = hist_lc(lags, lc_codes, 'all', lag_bin_bounds, density=density)
            ax.plot(bin_centres, hist, '--', color='k', ms=0, linewidth=1)

        ax.tick_params(labelsize=14)
        ax.set_xlim(lag_bin_bounds[[0, -1]])
        ax.set_xlabel('phase difference (days)', fontsize=16)

        if density:
            ax.set_ylabel('pdf', fontsize=16)
        else:
            ax.set_ylabel('number of pixels', fontsize=16)

        if show_95ci:
            median_lag_error = median_data[band]
            ax.plot([-median_lag_error, median_lag_error], [0.9, 0.9], "k|-",
                    transform=ax.get_xaxis_transform())
            ax.text(0.5, 0.95, '95% CI', fontsize=8,
                    transform=ax.transAxes, ha='center')

        ax.text(0.03, 0.9, f'({letter})', fontsize=14, fontweight="bold", transform=ax.transAxes)
        ax.text(0.03, 0.82, f'{band[0]}\u2013{band[1]} days', fontsize=14, transform=ax.transAxes)
        ax.axvline(0, color='gray', lw=0.5, alpha=0.3, zorder=0)

    labels = (
        ("density", density),
        ("median95ci", show_95ci),
        ("showall", all_lc_line),
    )

    if filename_out is None:
        label2 = "_".join(label for label, switch in labels if switch)
        label2 = "_" + label2 if label2 else label2
        filename = os.path.join(figures_dir,
                                f"land_cover_subplots_global{label2}.{plot_type}")
    else:
        filename = os.path.join(figures_dir, filename_out)

    plt.savefig(filename, dpi=600, bbox_inches='tight')


def plot_single_band_distribution(output_dirs, lag_data, median_data, band, lag_bin_bounds,
                                  density=False, show_95ci=True, all_lc_line=False,
                                  filename_out=None, plot_type="png"):

    lag_band = lag_data[band]

    figures_dir = output_dirs["figures"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    # plt.subplots_adjust(hspace=0.45, wspace=0.4)
    lc_codes = copernicus_land_cover(lat_south=-55, lat_north=55)
    lc_colors = {'Bare/sparse vegetation': '#ffd92f',
                 'Herbaceous vegetation': '#EA8294',
                 'Shrubland': '#AA4499',
                 'Cropland': '#88CCEE',
                 'Open forest': '#889933',
                 'Closed forest': '#117733'}

    bin_centres = lag_bin_bounds[:-1] + (lag_bin_bounds[1] - lag_bin_bounds[0])/2.

    for land_cover_code in lc_colors.keys():
        hist = hist_lc(lag_band, lc_codes, land_cover_code, lag_bin_bounds, density=density)
        ax.plot(bin_centres, hist, '-o', color=lc_colors[land_cover_code], ms=2.5, label=land_cover_code)

    if all_lc_line:
        hist = hist_lc(lag_band, lc_codes, 'all', lag_bin_bounds, density=density)
        ax.plot(bin_centres, hist, '--', color='k', ms=0, linewidth=1, label='All')
    ax.tick_params(labelsize=14)
    ax.set_xlim(lag_bin_bounds[[0, -1]])
    ax.set_xlabel('phase difference (days)', fontsize=16)
    if density:
        ax.set_ylabel('pdf', fontsize=16)
    else:
        ax.set_ylabel('number of pixels', fontsize=16)

    if show_95ci:
        median_lag_error = median_data[band]
        ax.plot([-median_lag_error, median_lag_error], [0.9, 0.9], "k|-",
                transform=ax.get_xaxis_transform())
        ax.text(0.5, 0.95, '95% CI', fontsize=8,
                transform=ax.transAxes, ha='center')

    ax.set_title(f'{band[0]}\u2013{band[1]} days', fontsize=14)
    ax.axvline(0, color='gray', lw=0.5, alpha=0.3, zorder=0)
    ax.legend(loc='upper left', fontsize=8, framealpha=1)

    labels = (
        ("density", density),
        ("median95ci", show_95ci),
        ("showall", all_lc_line),
    )

    if filename_out is None:
        label1 = f"{band[0]}-{band[1]}"
        label2 = "_".join(label for label, switch in labels if switch)
        label2 = "_" + label2 if label2 else label2
        filename = f"land_cover_lag_distribution_global_{label1}{label2}.{plot_type}"
        filename = os.path.join(figures_dir, filename)
    else:
        filename = os.path.join(figures_dir, filename_out)

    plt.savefig(filename, dpi=600, bbox_inches='tight')



def plot_global_percent_validity(output_dirs, ax, validity, num_obs):

    land_cover_codes = ['Bare/sparse vegetation','Herbaceous vegetation',
     'Shrubland','Cropland','Open forest', 'Closed forest']
    tile = 'global'

    seasons = validity.keys()
    tile_lat_south = lats_south[tile]
    tile_lat_north = lats_north[tile]
    lc_codes = copernicus_land_cover(lat_south=tile_lat_south, lat_north=tile_lat_north)
    coherent_pixels = {code: 0 for code in land_cover_codes}
    incoherent_pixels = {code: 0 for code in land_cover_codes}
    no_obs_pixels = {code: 0 for code in land_cover_codes}
    total_pixels = {code: 0 for code in land_cover_codes}
    inundation_masked_pixels = {code: 0 for code in land_cover_codes}
    for code in land_cover_codes:
        for season in seasons:
            coherent_pixels[code] += np.logical_and(validity[season]==1, lc_codes==code).sum()
            incoherent_pixels[code] += np.logical_and(validity[season]==0, lc_codes==code).sum()
            no_obs_pixels[code] += np.logical_and(validity[season]==2, lc_codes==code).sum()
            total_pixels[code] += (lc_codes==code).sum()

            possible_obs = num_obs[season]["num_total"]
            obs_before_mask = num_obs[season]["num_before"]
            obs_after_mask = num_obs[season]["num_after"]

            percent_before_mask = 100. * obs_before_mask / possible_obs
            percent_after_mask = 100. * obs_after_mask / possible_obs

            removed_by_inundation = np.logical_and(percent_before_mask>=30., percent_after_mask<30.)
            inundation_masked_pixels[code] += np.logical_and(removed_by_inundation==1, lc_codes==code).sum()

    coherent_list = np.array([value for value in coherent_pixels.values()])
    incoherent_list = np.array([value for value in incoherent_pixels.values()])
    inundation_masked_list = np.array([value for value in inundation_masked_pixels.values()])
    no_obs_list = np.array([value for value in no_obs_pixels.values()])
    no_obs_before_inundation = no_obs_list - inundation_masked_list
    total_list = np.array([value for value in total_pixels.values()])
    coherent_percent = 100.*coherent_list/total_list
    incoherent_percent = 100.*incoherent_list/total_list
    inundation_percent = 100.*inundation_masked_list/total_list
    no_obs_percent = 100.*no_obs_before_inundation/total_list
    coherent_percent_of_obs = 100.*(coherent_list/(coherent_list+incoherent_list))
    land_cover_labels = [line_break_string(label, 10) for label in land_cover_codes]
    width = 1
    lc_colors = ['#ffd92f', '#EA8294', '#AA4499', '#88CCEE', '#889933', '#117733']
    ax.bar(land_cover_labels, coherent_percent, width, color=lc_colors, edgecolor='k', linewidth=0.75, label='Coherency\n> 95% CL')
    ax.bar(land_cover_labels, incoherent_percent, width, color='#ffffff', edgecolor='k', linewidth=0.75, bottom=coherent_percent, label='No coherency\n> 95% CL')
    ax.bar(land_cover_labels, inundation_percent, width, color='#cccccc', edgecolor='k', hatch='/', linewidth=0.75, bottom=coherent_percent+incoherent_percent, label='Insufficient obs after inundation masking')
    ax.bar(land_cover_labels, no_obs_percent, width, color='#cccccc', edgecolor='k', linewidth=0.75, bottom=coherent_percent+incoherent_percent+inundation_percent, label='Insufficient obs')
    ax.set_ylim([0, 100])
    ax.set_xlim([-0.5, len(land_cover_codes)-0.5])
    ax.tick_params(labelsize=14)
    ax.tick_params(axis='x', labelrotation=90)
    for i in range(total_list.size):
        ax.text(i, 102, f'({total_list[i]})', fontsize=10, horizontalalignment='center')


def subplots_percent_validity(output_dirs, valid_data, num_data,
                              filename_out=None, plot_type="png"):

    figures_dir = output_dirs["figures"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    fig.subplots_adjust(right=0.8, bottom=0.25, wspace=0.05)

    for ax, letter, (band, valid) in zip([ax1, ax2], "ab", valid_data.items()):
        plot_global_percent_validity(output_dirs, ax, valid, num_data)
        ax.set_title(f"({letter}) {band[0]}\u2013{band[1]} days", fontsize=16, pad=25)

    color_list = ['#ffd92f', '#EA8294', '#AA4499', '#88CCEE', '#889933', '#117733']
    greys = ['#cccccc']*6
    whites = ['#ffffff']*6
    cmaps = [color_list, whites, greys, greys]
    cmap_labels = ['Coherency\n> 95% CL', 'No coherency\n> 95% CL', 'Insufficient obs\nafter inundation\nmasking', 'Insufficient obs']
    cmap_handles = [Rectangle((0, 0), 1, 1, edgecolor='k', linewidth=0.75) for _ in cmaps]
    handler_map = dict(zip(cmap_handles, 
                           [StripyPatch(cm) for cm in cmaps]))
    bar_legend = ax2.legend(handles=cmap_handles, labels=cmap_labels, handler_map=handler_map, fontsize=12,
                            loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=True)
    # this bit just uses a second legend to put nice borders around the legend patches
    # and hide artefacts in the non-striped boxes
    dummy_labels = ['\n', '\n', '\n\n', '']
    dummy_colours = ['none', '#ffffff', '#cccccc', '#cccccc']
    dummy_hatches = ['', '', '//', '']
    border_boxes = [Rectangle((0, 0), 1.05, 1.05, fc=fc, hatch=h, edgecolor='k', linewidth=0.5) for fc, h in zip(dummy_colours, dummy_hatches)]
    legend_borders = ax2.legend(handles=border_boxes, labels=dummy_labels,
                                fontsize=12, loc='center left', bbox_to_anchor=(1.05, 0.5),
                                frameon=False)
    ax2.add_artist(bar_legend)
    ax1.set_ylabel(r'% of pixels', fontsize=16)

    if filename_out is None:
        filename = os.path.join(figures_dir,
                                f'validity_percentage_by_land_cover_global_subplots_inundation.{plot_type}')
    else:
        filename = filename_out

    plt.savefig(filename, dpi=600)


def main():

    ###########################################################################
    # Parse command line args and load input file.
    ###########################################################################
    parser = ul.get_arg_parser()
    args = parser.parse_args()

    metadata = ul.load_yaml(args)
    metalags = metadata["lags"]

    output_dirs = metadata.get("output_dirs", None)

    datasets = metadata["datasets"]

    bands = [tuple(b) for b in
             ul.get_this_or_that("bands", metalags, metadata["filter"])
    ]

    seasons = metalags.get("seasons", None)
    lag_bin_bounds = {b: np.arange(*a) for b, a in zip(bands, metalags.get("lag_bin_bounds", None))}
    plot_type = metadata["plots"].get("type", "png")

    ul.check_dirs(output_dirs,
                  input_names=("spectra", "spectra_filtered","number_obs"),
                  output_names=("figures",))

    ###########################################################################
    # Run the analysis.
    ###########################################################################
    lag_data = {
        band: all_season_lags(output_dirs, datasets, 'global', *band, seasons=seasons) for band in bands
    }

    valid_data = {
        band: all_season_validity(output_dirs, datasets, 'global', *band, seasons=seasons) for band in bands
    }

    num_obs = load_num_obs(output_dirs["number_obs"], seasons)

    median_data = {
        band: median_95ci_width(output_dirs, datasets, 'global', *band, seasons=seasons) for band in bands
    }

    fig3_lag_bin_bounds = np.arange(-30, 31, 1)

    subplots(output_dirs, lag_data, median_data, fig3_lag_bin_bounds,
             density=True,
             show_95ci=True,
             all_lc_line=True,
             plot_type=plot_type)

    subplots_percent_validity(output_dirs, valid_data, num_obs,
                              plot_type=plot_type)

    for band in bands:
        filename_out = f"phase_{band[0]}-{band[1]}.{plot_type}"
        print(filename_out)
        plot_single_band_distribution(output_dirs, lag_data, median_data, band,
                                      lag_bin_bounds[band],
                                      density=True, show_95ci=True, all_lc_line=False,
                                      filename_out=filename_out, plot_type=plot_type)


if __name__ == '__main__':
    main()
