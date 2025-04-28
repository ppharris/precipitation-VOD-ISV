import matplotlib.pyplot as plt
import numpy as np
import os

from ancils.aridity import load_aridity, standard_mapping, standard_colors
import utils.load as ul


def process_data(lag_data_dir, bands, seasons, classes):

    land_cover, land_classes = classes
    print(land_classes)

    data = {}
    for season in seasons:
        # Initialise storage for mean lags.
        bdata = {lc:[] for lc in land_classes}
        for band in bands:
            # Load the lag map data for this band.
            band_lower, band_upper = band
            fname = f"lag_{season}_{band_lower}-{band_upper}.npy"
            lag = np.load(os.path.join(lag_data_dir, fname))

            # Calculate the mean lag for each land cover class.
            for land_class in land_classes:
                llag = np.where(land_cover == land_class.value, lag, np.nan)
                lag_mean = np.nanmean(llag)
                nlags = np.sum(~np.isnan(llag))
                lag_sem = np.nanstd(llag) / np.sqrt(nlags)
                bdata[land_class].append((lag_mean, lag_sem))

                print(f"{season}: {band}: {land_class.name}: {lag_mean:5.1f}, {lag_sem:5.2f}, {nlags}")

        data[season] = bdata

    return data


def make_plots(output_dirs, bands, seasons, classes, plot_type="png"):

    figures_dir = output_dirs["figures"]

    lag_data_dir = output_dirs["lag_data"]
    data = process_data(lag_data_dir, bands, seasons, classes)

    F, axs = plt.subplots(nrows=1, ncols=len(seasons), squeeze=False)

    for ax, season in zip(axs.flat, seasons):
        sdata = data[season]

        idx = range(len(bands))

        for land_class, data in sdata.items():
            lags, sems = (d for d in zip(*data))
            ax.errorbar(idx, lags, yerr=sems,
                        marker="o", ls="-", lw=2,
                        elinewidth=1, capsize=3,
                        color=standard_colors[land_class],
                        label=land_class.name)

        ax.legend()
        ax.set_xlabel("Period band (days)")
        ax.set_xticks(ticks=idx, labels=(f"{bl}-{bu}" for (bl, bu) in bands))
        ax.axhline(0, lw=1, color="k", alpha=0.2)

        ax.set_ylabel("Phase lag (days)")

    filename = os.path.join(figures_dir, f"lag_by_band_aridity_plot.{plot_type}")
    plt.savefig(filename, dpi=600, bbox_inches='tight')

    return


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
    # Load a map of aridity index.
    ###########################################################################
    aridity = load_aridity(mapping=standard_mapping,
                           intersection={"latitude":(-55, 55)})

    if aridity.min() == 0:
        print("Warning: Some values are underbin.")
    if aridity.max() > len(standard_mapping):
        print("Warning: Some values are overbin.")

    ###########################################################################
    # Run the analysis.
    ###########################################################################
    make_plots(output_dirs, bands, seasons, (aridity, standard_mapping),
               plot_type=plot_type)

    return


if __name__ == '__main__':
    main()
