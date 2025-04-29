
class Dataset(object):
    """Class encapsulating metadata for an input dataset."""

    def __init__(self, name, varname, units, path_pattern=None, path_getter=None):
        """Initialise metadata for a dataset.

        Args:
            name : str
                Short name. Often used in input/output filenames, so best avoid
                spaces.
            varname : str
                NetCDF variable name.
            units : str
                Variable units string used for Iris cubes and netCDF attributes.
            path_pattern : str, optional
                String used to generate an input file path getter function using
                a standard _path_getter() function.  Must be specified if a
                bespoke function is not provided via the path_getter arg, and is
                ignored if path_getter arg is provided.
            path_getter : str, optional
                Function that returns an input file path.  Use this if your
                input paths can't be generated just by substituting the year
                into a string.
        """
        self.name = name
        self.varname = varname
        self.units = units

        if path_getter is not None:
            self.get_path = path_getter
        elif path_pattern is not None:
            self.get_path = _path_getter(path_pattern)
        else:
            raise ValueError("Either path_pattern or path_getter must be specified.")


def _path_getter(path_pattern):
    """Generate a function that returns a file path.

    Args:
        path_pattern : str
            A string that must contain at least one instance of the partial
            format string "{year".

    Returns:
        get_path : function
            A function that returns a path name based on an input year.
    """
    if "{year" not in path_pattern:
        raise ValueError("path_pattern must contain the string '{year}'.")

    def get_path(year):
        return path_pattern.format(year=year)

    return get_path


###############################################################################
# Instances of datasets used in this analysis.
###############################################################################
IMERG = Dataset("IMERG", "precipitationCal", "mm d-1",
                path_pattern="/localscratch/wllf029/bethar/IMERG/IMERG.V06.{year}.daily.nc4")

IMERG_RG = Dataset("IMERG-RG", "precipitationCal", "mm d-1",
                   path_pattern="/prj/nceo/bethar/IMERG/regrid_p25_global/IMERG.V06.{year}.daily_p25.nc")

VOD = Dataset("VOD", "vod", "1",
              path_pattern="/prj/nceo/bethar/VODCA_global/filtered/X-band/VOD-X-band_filtered_{year}.nc")

VOD_SW = Dataset("VOD-SW", "vod", "1",
                 path_pattern="/prj/nceo/bethar/VODCA_global/filtered/filtered_surface_water/X-band/VOD-X-band_filtered_surface_water_{year}.nc")

CCI_SM = Dataset("CCI-SM", "sm", "m3 m-3",
                 path_pattern="/prj/swift/ESA_CCI_SM/year_files_v6.1_combined_GLOBAL/{year}_volumetric_soil_moisture_daily.nc")

SWAMPS = Dataset("SWAMPS", "frac_surface_water", "1",
                 path_pattern="/prj/nceo/bethar/SWAMPS_daily/SWAMPS-{year}.nc")

NDVI_AQUA = Dataset("NDVI-AQUA", "NDVI", "1",
                    path_pattern="/prj/nceo/bethar/MODIS-NDVI-16day/modis_aqua_16-day_ndvi_0p25_{year}.nc")

NDVI_TERRA = Dataset("NDVI-TERRA", "NDVI", "1",
                    path_pattern="/prj/nceo/bethar/MODIS-NDVI-16day/modis_terra_16-day_ndvi_0p25_{year}.nc")

FLUXCOM_JRA = Dataset("FLUXCOM-JRA-025", "GPP", "g m-2 d-1",
                      path_pattern="/prj/nceo/ppha/cpeo/data/fluxcom/GPP.RS_METEO.FP-ALL.MLM-ALL.METEO-CRUJRA_v1.720_360.daily.{year}.nc")

FLUXCOM_ERA = Dataset("FLUXCOM-ERA-025", "GPP", "g m-2 d-1",
                      path_pattern="/prj/nceo/ppha/cpeo/data/fluxcom/GPP.RS_METEO.FP-ALL.MLM-ALL.METEO-ERA5.720_360.daily.{year}.nc")


def _initialise_datasets():
    """Generate a function that returns a dataset based on name."""
    all_datasets = {
        d.name: d for d in (
            IMERG, IMERG_RG,
            VOD, VOD_SW,
            CCI_SM,
            SWAMPS,
            NDVI_AQUA, NDVI_TERRA,
            FLUXCOM_JRA, FLUXCOM_ERA,
        )
    }

    def get_dataset(name):
        return all_datasets.get(name, None)

    return get_dataset

get_dataset = _initialise_datasets()


if __name__ == "__main__":
    pass

