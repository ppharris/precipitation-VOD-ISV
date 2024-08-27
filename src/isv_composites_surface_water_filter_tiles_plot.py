import matplotlib.pyplot as plt
import numpy as np
import os
import string

import utils_load as ul


def land_cover_full_name(land_cover):
    full_names = {'baresparse': 'Bare/sparse vegetation',
                  'shrub': 'Shrubland',
                  'herb': 'Herbaceous vegetation',
                  'crop': 'Cropland',
                  'openforest': 'Open forest',
                  'closedforest': 'Closed forest'}
    return full_names[land_cover]


def plot_composites(output_dirs, land_covers, days_range=60, ndvi=False):

    data_comp_dir = output_dirs["data_isv_comp"]
    figures_dir = output_dirs["figures"]

    fig, axs = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(9, 8))
    axlist = axs.flatten()
    blue =  '#56B4E9'
    green = '#009E73'
    brown = '#E69F00'
    light_green = '#88F74D'
    alphabet = string.ascii_lowercase
    days_around = np.arange(-days_range, days_range+1)
    for i, land_cover in enumerate(land_covers):
        ax = axlist[i]

        imerg_composite = np.load(os.path.join(data_comp_dir, f'imerg_composite_global_55NS_{land_cover}.npy'))
        vod_composite = np.load(os.path.join(data_comp_dir, f'vod_composite_global_55NS_{land_cover}.npy'))
        sm_composite = np.load(os.path.join(data_comp_dir, f'sm_composite_global_55NS_{land_cover}.npy'))
        n = np.load(os.path.join(data_comp_dir, f'n_global_55NS_{land_cover}.npy'))

        ax.plot(days_around, imerg_composite, color=blue, label='precipitation')
        ax.plot(days_around, sm_composite, color=brown, label='SSM')
        ax.plot(days_around, vod_composite, color=green, label='VOD')
        if ndvi:
            ndvi_composite = np.load(os.path.join(data_comp_dir, f'ndvi_composite_global_55NS_{land_cover}.npy'))
            ax.plot(days_around, ndvi_composite, '--', color=light_green, label='NDVI')
        ax.set_ylim([-0.5, 1.75])
        ax.set_xticks(np.arange(-60, 61, 5), minor=True)
        ax.set_xticks(np.arange(-60, 61, 30))
        ax.set_yticks(np.arange(-0.5, 2., 0.5))
        ax.axhline(0, color='gray', alpha=0.3, zorder=0)
        ax.axvline(0, color='gray', alpha=0.3, zorder=0)
        ax.set_xlim([-days_range, days_range])
        ax.set_xlabel(f"days since intraseasonal\nprecipitation maximum", fontsize=14)
        ax.set_ylabel("standardised anomaly", fontsize=13)
        ax.label_outer()
        ax.tick_params(labelsize=12)
        if i == 0:
            ax.legend(loc='upper left', fontsize=11)
        ax.set_title(f'$\\bf{{({alphabet[i]})}}$ {land_cover_full_name(land_cover)}', fontsize=12)
        ax.text(58, 1.68, f'mean(n) = {int(np.round(np.mean(n)))}\nmin(n) = {int(np.min(n))}', ha='right', va='top')
    plt.tight_layout()

    save_filename = os.path.join(figures_dir, 'vod_around_precip_isv_maxima_lowpass_1std_norm_withsm_subplots_global_55NS')
    if ndvi:
        save_filename += '_ndvi_dashed'
    plt.savefig(f'{save_filename}.png', dpi=400)
    plt.savefig(f'{save_filename}.pdf', dpi=400)
    plt.savefig(f'{save_filename}.eps', dpi=400)


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
    ndvi = metadata["isv"].get("ndvi", None)

    ###########################################################################
    # Run the analysis.
    ###########################################################################
    plot_composites(output_dirs, land_covers, days_range=days_range, ndvi=ndvi)


if __name__ == '__main__':
    main()
