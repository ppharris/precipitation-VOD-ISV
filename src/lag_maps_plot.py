import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from itertools import product
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np
import os

import utils.load as ul


def to_percent(numer, denom):
    try:
        result = numer * 100.0 / denom
    except ZeroDivisionError:
        result = np.nan
    return result


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

    fname_out = os.path.join(figures_dir, f"lag_maps_with95ci.{plot_type}")
    plt.savefig(fname_out, dpi=600, bbox_inches='tight')


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
