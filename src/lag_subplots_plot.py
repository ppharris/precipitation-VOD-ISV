import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from itertools import product
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import os

import utils_load as ul


def to_percent(numer, denom):
    try:
        result = numer * 100.0 / denom
    except ZeroDivisionError:
        result = np.nan
    return result


def global_plots_mean_estimate(output_dirs, plot_type="png"):

    lag_data_dir = output_dirs["lag_data"]
    figures_dir = output_dirs["figures"]

    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes,
                  dict(projection=projection))
    lons = np.arange(-180, 180, 0.25) + 0.5*0.25
    lats = np.arange(-55, 55, 0.25) + 0.5*0.25
    lon_bounds = np.hstack((lons - 0.5*0.25, np.array([lons[-1]+0.5*0.25])))
    lat_bounds = np.hstack((lats - 0.5*0.25, np.array([lats[-1]+0.5*0.25])))
    fig = plt.figure(figsize=(16, 10))
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(4, 2),
                    axes_pad=0.2,
                    cbar_location='bottom',
                    cbar_mode='single',
                    cbar_pad=0.15,
                    cbar_size='10%',
                    label_mode='')  # note the empty label_mode

    seasons = np.repeat(['MAM', 'JJA', 'SON', 'DJF'], 2)
    band_days_lower = [25, 40]*4
    band_days_upper = [40, 60]*4

    contour_multiple = 5.
    lag_contour_levels = np.arange(-30., 31., contour_multiple).astype(int)
    colormap = cm.get_cmap('RdYlBu_r', 2*(lag_contour_levels.size+7))
    colors = list(colormap(np.arange(2*(lag_contour_levels.size+7))))
    colors_to_take = [0, 1, 2, 4, 7, 12, 27, 32,35, 37, 38, 39] 
    no_centre_colors = [colors[i] for i in colors_to_take]

    cmap = mpl.colors.ListedColormap(no_centre_colors, "")
    norm = mpl.colors.BoundaryNorm(lag_contour_levels, ncolors=len(lag_contour_levels)-1, clip=False)
    cmap.set_bad('#e7e7e7')
    cmap.set_under('white')

    for i, ax in enumerate(axgr):
        lag = np.load(os.path.join(lag_data_dir, f'lag_{seasons[i]}_{band_days_lower[i]}-{band_days_upper[i]}.npy'))
        lag_for_hist = np.copy(lag)
        total_lags = (~np.isnan(lag_for_hist)).sum()
        neg_lags = (lag_for_hist < 0.).sum()
        pos_lags = (lag_for_hist > 0.).sum()
        percent_neg = np.round(float(neg_lags)/float(total_lags) * 100.)
        percent_pos = np.round(float(pos_lags)/float(total_lags) * 100.)
        no_csa = np.load(os.path.join(lag_data_dir, f'no_csa_{seasons[i]}_{band_days_lower[i]}-{band_days_upper[i]}.npy'))
        invalid_but_csa = np.logical_and(~no_csa, np.isnan(lag))
        lag[invalid_but_csa] = -999
        ax.coastlines(color='k',linewidth=0.2)
        ax.text(0.015, 0.825, f'{seasons[i]}', fontsize=16, transform=ax.transAxes)
        p = ax.pcolormesh(lon_bounds, lat_bounds, lag, transform=ccrs.PlateCarree(), 
                          cmap=cmap, norm=norm, rasterized=True)
        ax.set_extent((-180, 180, -55, 55), crs=ccrs.PlateCarree())
        ax.set_xticks(np.arange(-90, 91, 90), crs=projection)
        ax.set_yticks(np.arange(-50, 51, 50), crs=projection)
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.tick_params(labelsize=14)
        ax.tick_params(axis='x', pad=5)
        ax_sub = inset_axes(ax, width=1,
                       height=0.7, loc='lower left',
                       bbox_to_anchor=(-175, -50),
                       bbox_transform=ax.transData,
                       borderpad=0)
        N, bins, patches = ax_sub.hist(lag_for_hist.ravel(), bins=lag_contour_levels)
        for i in range(len(patches)):
            patches[i].set_facecolor(no_centre_colors[i])
        ax_sub.set_yticks([])
        ax_sub.set_xticks([])
        ax_sub.patch.set_alpha(0.)
        ax_sub.spines['right'].set_visible(False)
        ax_sub.spines['left'].set_visible(False)
        ax_sub.spines['top'].set_visible(False)
        # add detail on percent pos/neg
        ymin, ymax = ax_sub.get_ylim()
        ax_sub.set_ylim(top=1.3 * ymax)
        ax_sub.axvline(0, color='k', linestyle='--', linewidth=0.5, dashes=(5, 5))
        ax_sub.text(0.55, 0.82, f'{int(percent_pos):d}%', transform=ax_sub.transAxes, horizontalalignment='left', fontsize=10)
        ax_sub.text(0.45, 0.82, f'{int(percent_neg):d}%', transform=ax_sub.transAxes, horizontalalignment='right', fontsize=10)
    axes = np.reshape(axgr, axgr.get_geometry())
    for ax in axes[:-1, :].flatten():
        ax.xaxis.set_tick_params(which='both', 
                                 labelbottom=False, labeltop=False)
    for ax in axes[:, 1:].flatten():
        ax.yaxis.set_tick_params(which='both', 
                                 labelbottom=False, labeltop=False)
    axes = np.reshape(axgr, axgr.get_geometry())
    axes[0, 0].set_title(u"25\u201340 days", fontsize=18)
    axes[0, 1].set_title(u"40\u201360 days", fontsize=18)    
    cbar = axgr.cbar_axes[0].colorbar(p)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.set_xlabel('phase difference (days)', fontsize=18)

    filename = os.path.join(figures_dir,
                            f"lag_subplots_mean_phase_diff_estimate.{plot_type}")
    plt.savefig(filename, dpi=600, bbox_inches='tight')


def global_plots_with95ci(output_dirs, bands, seasons, plot_raw_lags=False, plot_type="png"):

    lag_data_dir = output_dirs["lag_data"]
    figures_dir = output_dirs["figures"]

    if plot_raw_lags:
        cmap = mpl.colormaps.get_cmap("RdYlBu_r")
        negative_colour = cmap(0)
        unsure_colour = "#E1BE6A"
        positive_colour = cmap(255)
        norm = None

        kw_axesgrid = {
            "cbar_location": "right",
            "cbar_mode": "each",
            "cbar_size": "2%",
            "cbar_pad": 0.25,
        }
    else:
        positive_colour = '#B0154B'
        unsure_colour = '#E1BE6A'
        negative_colour = '#6072C1'
        lag_colours = [negative_colour, unsure_colour, positive_colour]
        cmap = mpl.colors.ListedColormap(lag_colours, "")
        cmap.set_bad('white')
        cmap.set_under('#c7c7c7')
        norm = mpl.colors.BoundaryNorm(np.arange(4)-0.5, ncolors=3, clip=False)

        kw_axesgrid = {
            "cbar_location": "bottom",
            "cbar_mode": "single",
            "cbar_size": "10%",
            "cbar_pad": 0.15,
        }

    lons = np.arange(-180, 180, 0.25) + 0.5*0.25
    lats = np.arange(-55, 55, 0.25) + 0.5*0.25
    lon_bounds = np.hstack((lons - 0.5*0.25, np.array([lons[-1]+0.5*0.25])))
    lat_bounds = np.hstack((lats - 0.5*0.25, np.array([lats[-1]+0.5*0.25])))

    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes,
                  dict(projection=projection))

    fig = plt.figure(figsize=(16, 10))
    axgr = AxesGrid(fig, 111,
                    axes_class=axes_class,
                    nrows_ncols=(len(seasons), len(bands)),
                    axes_pad=0.2,
                    label_mode="keep",
                    **kw_axesgrid)

    for ax, cax, (season, (band_lower, band_upper)) in zip(axgr, axgr.cbar_axes, product(seasons, bands)):

        file_lag = os.path.join(lag_data_dir, f'lag_{season}_{band_lower}-{band_upper}.npy')
        file_lag_error = os.path.join(lag_data_dir, f'lag_error_{season}_{band_lower}-{band_upper}.npy')
        file_no_csa = os.path.join(lag_data_dir, f'no_csa_{season}_{band_lower}-{band_upper}.npy')

        lag = np.load(file_lag)
        lag_error = np.load(file_lag_error)
        no_csa = np.load(file_no_csa)

        lag_upper = lag + lag_error
        lag_lower = lag - lag_error

        positive_confidence_interval = (lag_lower > 0.)
        negative_confidence_interval = (lag_upper < 0.)
        confidence_interval_overlaps_zero = (np.sign(lag_upper)/np.sign(lag_lower) == -1)
        invalid_but_csa = np.logical_and(~no_csa, np.isnan(lag))

        total_lags = (~np.isnan(lag)).sum()

        percent_neg = to_percent(negative_confidence_interval.sum(), total_lags)
        percent_pos = to_percent(positive_confidence_interval.sum(), total_lags)
        percent_unsure = to_percent(confidence_interval_overlaps_zero.sum(), total_lags)

        if plot_raw_lags:
            lag_plot = lag
        else:
            lag_plot = np.ones_like(lag) * np.nan
            lag_plot[positive_confidence_interval] = 2
            lag_plot[negative_confidence_interval] = 0
            lag_plot[confidence_interval_overlaps_zero] = 1
            lag_plot[invalid_but_csa] = -999

        p = ax.pcolormesh(lon_bounds, lat_bounds, lag_plot,
                          cmap=cmap, norm=norm, rasterized=True)

        ax.coastlines(color='k',linewidth=0.2)
        ax.set_extent((-180, 180, -55, 55), crs=projection)
        ax.set_xticks(np.arange(-90, 91, 90), crs=projection)
        ax.set_yticks(np.arange(-50, 51, 50), crs=projection)
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.tick_params(labelsize=14)
        ax.tick_params(axis='x', pad=5)

        ax.set_title(f"{band_lower}–{band_upper} days", fontsize=11)
        ax.text(0.015, 0.225, f'{season}', fontsize=11, transform=ax.transAxes)
        ax.text(0.05, 0.05, f'{percent_neg:3.0f}%', color=negative_colour, transform=ax.transAxes, horizontalalignment='center', fontsize=12)
        ax.text(0.13, 0.05, f'{percent_unsure:3.0f}%', color='#ba8e25', transform=ax.transAxes, horizontalalignment='center', fontsize=12)
        ax.text(0.21, 0.05, f'{percent_pos:3.0f}%', color=positive_colour, transform=ax.transAxes, horizontalalignment='center', fontsize=12)

        # Separate colorbar on every map because the ranges may be different.
        if plot_raw_lags:
            cbar = cax.colorbar(p)

    axes = np.reshape(axgr, axgr.get_geometry())
    for ax in axes[:-1, :].flatten():
        ax.xaxis.set_tick_params(which='both', 
                                 labelbottom=False, labeltop=False)
    for ax in axes[:, 1:].flatten():
        ax.yaxis.set_tick_params(which='both', 
                                 labelbottom=False, labeltop=False)

    # Single colorbar for all plots because it's categories.
    if not plot_raw_lags:
        cbar = axgr.cbar_axes[0].colorbar(p, ticks=[0, 1, 2])
        cbar.ax.set_xticklabels(['negative\nphase difference', 
                                 'phase difference\nindistinguishable from zero',
                                 'positive\nphase difference'])
        cbar.ax.tick_params(labelsize=16)

    fname_out = os.path.join(figures_dir, f"lag_subplots_with95ci.{plot_type}")
    plt.savefig(fname_out, dpi=600, bbox_inches='tight')


def lag_sign_stats(output_dirs, season, band_days_lower, band_days_upper):

    lag_data_dir = output_dirs["lag_data"]

    lag = np.load(os.path.join(lag_data_dir, f'lag_{season}_{band_days_lower}-{band_days_upper}.npy'))
    lag_sign = np.ones_like(lag) * np.nan
    lag_error = np.load(os.path.join(lag_data_dir, f'lag_error_{season}_{band_days_lower}-{band_days_upper}.npy'))
    lag_lower = lag - lag_error
    lag_upper = lag + lag_error
    total_px = (~np.isnan(lag)).sum()
    pos_px = (lag > 0.).sum()
    neg_px = (lag < 0.).sum()
    pos_less_7 = np.logical_and(lag>0., lag<7.).sum()
    pos_less_10 = np.logical_and(lag>0., lag<10.).sum()
    pos_ci_px = (lag_lower>0.).sum()
    neg_ci_px = (lag_upper<0.).sum()
    cross_ci = np.logical_and(lag_upper>0., lag_lower<0.)
    cross_ci_px = (cross_ci).sum()
    print(f'total pixels: {total_px}')
    print(f'positive lag: {pos_px}')
    print(f'negative lag: {neg_px}')
    print(f'positive and less than 7: {pos_less_7}')
    print(f'positive and less than 10: {pos_less_10}')
    print('Accounting for 95% CI:')
    print(f'positive: {pos_ci_px}')
    print(f'negative: {neg_ci_px}')
    print(f'sign uncertain: {cross_ci_px}')


def main():

    ###########################################################################
    # Parse command line args and load input file.
    ###########################################################################
    parser = ul.get_arg_parser()
    args = parser.parse_args()

    metadata = ul.load_yaml(args)

    output_dirs = metadata.get("output_dirs", None)
    bands = [tuple(b) for b in metadata["lags"].get("bands", None)]
    seasons = metadata["lags"].get("seasons", None)
    plot_type = metadata["plots"].get("type", "png")

    ul.check_dirs(output_dirs,
                  input_names=("lag_data",),
                  output_names=("figures",))

    ###########################################################################
    # Run the analysis.
    ###########################################################################
    global_plots_with95ci(output_dirs, bands, seasons, plot_type=plot_type)


if __name__ == '__main__':
    main()
