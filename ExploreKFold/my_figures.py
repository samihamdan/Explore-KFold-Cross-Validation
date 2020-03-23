import re
import colorcet
import holoviews as hv
from bokeh.colors import RGB
from bokeh.models import HoverTool

hv.extension('bokeh')


def hex_to_rgb(hex_string, alpha):
    hex_rgb = re.findall('..', hex_string[1:])
    return RGB(*[int(hex_val, 16) for hex_val in hex_rgb], a=alpha)


def my_fold_colors(ds):
    available_colors = hv.Cycle(colorcet.glasbey).values

    while len(available_colors) < len(ds.data.data_split.unique()):
        available_colors = available_colors + available_colors

    return available_colors


def create_scatter(ds, fold):
    color_palette = my_fold_colors(ds)

    def fold_to_nbr(fold_name):
        return int(''.join([s for s in fold_name if s.isdigit()]))

    ds.data['hover_fold'] = ds.data.apply(lambda row: ('in fold ' if row['in_train_set'] else 'out of fold ')
                                                      + str(fold_to_nbr(row['data_split'])), axis=1)

    hover = HoverTool(tooltips=[
        ("fold", "@hover_fold"),
        ("x,y, y_pred", "@x, @y, @y_pred")])

    if fold == 'all':
        ds.data['color'] = ds.data.data_split.map(lambda fold_name: color_palette[fold_to_nbr(fold_name)])
        ds_filtered = ds.select(in_train_set=['out_fold'])
    else:
        ds.data['color'] = ds.data.in_train_set.map(lambda in_train: 'black' if in_train else 'lightgrey')
        ds_filtered = ds.select(in_train_set=['in_fold', 'out_fold'], data_split=fold)

    return (ds_filtered
            .to(hv.Points, kdims=['x', 'y'], vdims=['color', 'hover_fold', 'y_pred'], groupby=[])
            .opts(color='color', size=5, width=800, height=550)
            .opts('Points', tools=[hover]))
