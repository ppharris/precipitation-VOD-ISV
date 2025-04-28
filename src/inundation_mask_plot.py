import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import iris
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np
import os

import utils.load as ul


def removed_for_inundation(number_obs_dir, season):
    total_possible_obs = np.load(os.path.join(number_obs_dir, f'total_possible_obs_{season}.npy'))
    total_obs_no_sw_mask = np.load(os.path.join(number_obs_dir, f'total_obs_no_sw_mask_{season}.npy'))
    total_obs_sw_mask = np.load(os.path.join(number_obs_dir, f'total_obs_sw_mask_{season}.npy'))
    percent_before_mask = 100. * total_obs_no_sw_mask / total_possible_obs
    percent_after_mask = 100. * total_obs_sw_mask / total_possible_obs
    removed_by_mask = np.logical_and(percent_before_mask>=30., percent_after_mask<30.)
    return removed_by_mask


def inundation_mask_maps(output_dirs, seasons, plot_type="png"):

    number_obs_dir = output_dirs["number_obs"]
    figures_dir = output_dirs["figures"]

    cci_sm_mask = '/prj/nceo/bethar/VODCA_global/ESA-CCI-SOILMOISTURE-LAND_AND_RAINFOREST_MASK-fv04.2.nc' # this file is included in VODCA dataset
    lat55 = iris.Constraint(latitude=lambda cell: -55. <= cell.point <= 55.)
    land = np.flipud(iris.load_cube(cci_sm_mask, 'land').extract(lat55).data).astype(float)
    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes,
                  dict(map_projection=projection))
    lons = np.arange(-180, 180, 0.25) + 0.5*0.25
    lats = np.arange(-55, 55, 0.25) + 0.5*0.25
    lon_bounds = np.hstack((lons - 0.5*0.25, np.array([lons[-1]+0.5*0.25])))
    lat_bounds = np.hstack((lats - 0.5*0.25, np.array([lats[-1]+0.5*0.25])))

    ns = len(seasons)
    nrows_ncols = (ns//2 + ns % 2, 2)

    fig = plt.figure(figsize=(16, 8))
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=nrows_ncols,
                    axes_pad=0.2,
                    cbar_location='bottom',
                    cbar_mode='single',
                    cbar_pad=0.15,
                    cbar_size='10%',
                    label_mode='')  # note the empty label_mode

    for ax, season in zip(axgr, seasons):
        print(ax)
        obs_removed = removed_for_inundation(number_obs_dir, season).astype(float)
        ssm_obs = np.load(os.path.join(number_obs_dir, f'total_obs_ssm_{season}.npy'))
        total_days = np.load(os.path.join(number_obs_dir, f'total_possible_obs_{season}.npy'))
        ssm_obs_pc = 100. * ssm_obs / total_days
        ssm_obs_pc[land==0.] = np.nan
        obs_removed[obs_removed==0.] = np.nan
        obs_removed[np.logical_and(obs_removed==1, ssm_obs_pc==0)] = 2
        ax.coastlines(color='#999999',linewidth=0.1)
        ax.text(0.015, 0.825, season, fontsize=16, transform=ax.transAxes)
        p = ax.pcolormesh(lon_bounds, lat_bounds, obs_removed, transform=ccrs.PlateCarree(),
                          cmap=mpl.colors.ListedColormap(['#007dea', '#b2b2b2']), rasterized=True)
        ax.set_extent((-180, 180, -55, 55), crs=ccrs.PlateCarree())
        ax.set_xticks(np.arange(-90, 91, 90), crs=projection)
        ax.set_yticks(np.arange(-50, 51, 50), crs=projection)
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.tick_params(labelsize=14)
        ax.tick_params(axis='x', pad=5)
    axes = np.reshape(axgr, axgr.get_geometry())
    for ax in axes[:-1, :].flatten():
        ax.xaxis.set_tick_params(which='both',
                                 labelbottom=False, labeltop=False)
    for ax in axes[:, 1:].flatten():
        ax.yaxis.set_tick_params(which='both',
                                 labelbottom=False, labeltop=False)
    axes = np.reshape(axgr, axgr.get_geometry())
    cbar = axgr.cbar_axes[0].colorbar(p)
    cbar.set_ticks([1.25, 1.75])
    cbar.set_ticklabels(['masked', 'masked and no SSM obs'])
    cbar.ax.tick_params(labelsize=16)

    filename = os.path.join(figures_dir,
                            f"pixels_removed_by_inundation_mask.{plot_type}")
    plt.savefig(filename, dpi=1000, bbox_inches='tight')


def main():

    ###########################################################################
    # Parse command line args and load input file.
    ###########################################################################
    parser = ul.get_arg_parser()
    args = parser.parse_args()

    metadata = ul.load_yaml(args)

    output_dirs = metadata.get("output_dirs", None)
    seasons = metadata["lags"].get("seasons", None)
    plot_type = metadata["plots"].get("type", "png")

    ul.check_dirs(output_dirs,
                  input_names=("number_obs",),
                  output_names=("figures",)
    )

    ###########################################################################
    # Run the analysis.
    ###########################################################################
    inundation_mask_maps(output_dirs, seasons, plot_type=plot_type)

    return


if __name__ == "__main__":
    main()
