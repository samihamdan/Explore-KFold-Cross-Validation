import re
import colorcet
import holoviews as hv
from bokeh.colors import RGB

hv.extension('bokeh')


def hex_to_rgb(hex_string, alpha):
    hex_rgb = re.findall('..', hex_string[1:])
    return RGB(*[int(hex_val, 16) for hex_val in hex_rgb], a=alpha)


def my_fold_colors(ds):
    available_colors = hv.Cycle(colorcet.glasbey).values

    while len(available_colors) < len(ds.data.data_split.unique()):
        available_colors = available_colors + available_colors

    return available_colors
