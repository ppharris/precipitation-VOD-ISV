from enum import Enum
import iris
import numpy as np


class standard_classes(Enum):
    sparse_veg = 1
    herbaceous = 2
    shrubland = 3
    cropland = 4
    open_forest = 5
    closed_forest = 6


standard_mapping = {
    standard_classes.sparse_veg: [60, ],
    standard_classes.herbaceous: [20, ],
    standard_classes.shrubland: [30, ],
    standard_classes.cropland: [40, ],
    standard_classes.open_forest: [121, 122, 123, 124, 125, 126, ],
    standard_classes.closed_forest: [111, 112, 113, 114, 115, 116, ],
}


standard_colors = {
    standard_classes.sparse_veg: "#ffd92f",
    standard_classes.herbaceous: "#EA8294",
    standard_classes.shrubland: "#AA4499",
    standard_classes.cropland: "#88CCEE",
    standard_classes.open_forest: "#889933",
    standard_classes.closed_forest: "#117733",
}


def load_land_cover(mapping=None, intersection=None):

    land_cover = iris.load_cube("/prj/nceo/bethar/copernicus_landcover_2018.nc")

    if intersection is not None:
        land_cover = land_cover.intersection(**intersection)

    land_cover_data = land_cover.data.filled(0)

    if mapping is None:
        land_cover_out = land_cover_data
    else:
        land_cover_out = np.full_like(land_cover_data, 0)
        for new_class, src_classes in mapping.items():
            for src_class in src_classes:
                mask = land_cover_data == src_class
                land_cover_out[mask] = new_class.value

    return land_cover_out

