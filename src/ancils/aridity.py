"""The Global Aridity Index and Potential Evapotranspiration Database -
Version 3 (Global-AI_PET_v3)

Zomer et al (2022), Scientific Data, doi:10.1038/s41597-022-01493-1
Data doi:10.6084/m9.figshare.7504448.v5

AI = P/PET

Where P = mean annual precip
PET = mean annual reference crop evaporation.
"""

from enum import Enum
import iris
import numpy as np


class standard_classes(Enum):
    hyper_arid = 1
    arid = 2
    semi_arid = 3
    dry_sub_humid = 4
    humid = 5


standard_mapping = {
    standard_classes.hyper_arid: (0.0, 0.03),
    standard_classes.arid: (0.03, 0.20),
    standard_classes.semi_arid: (0.20, 0.50),
    standard_classes.dry_sub_humid: (0.50, 0.65),
    standard_classes.humid: (0.65, 10.0),
}

standard_colors = {
    standard_classes.hyper_arid: "tab:red",
    standard_classes.arid: "tab:orange",
    standard_classes.semi_arid: "tab:olive",
    standard_classes.dry_sub_humid: "tab:cyan",
    standard_classes.humid: "tab:blue",
}


def load_aridity(mapping=None, intersection=None, return_cube=False):
    file_ai = "/prj/nceo/bethar/Global-AI_ET0_v3_annual/global_aridity_index_annual_0pt25deg.nc"
    aridity = iris.load_cube(file_ai)

    if intersection is not None:
        aridity = aridity.intersection(**intersection)

    aridity_data = aridity.data.filled(np.nan)
        
    if mapping is None:
        aridity_out = aridity_data
    else:
        aridity_bins = np.unique([v for v in mapping.values()])
        aridity_out = np.digitize(aridity_data, bins=aridity_bins)

    if return_cube:
        aridity_out = aridity.copy(data=aridity_out)
        
    return aridity_out
