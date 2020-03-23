import re
import colorcet
import pandas as pd
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

    ds.data['hover_fold'] = ds.data.apply(lambda row: ('in fold ' if row['in_train_set'] == 'in_fold'
                                                       else 'out of fold ' if row['in_train_set'] == 'out_fold'
                                                       else 'extra test set')
                                                      + str(fold_to_nbr(row['data_split'])), axis=1)

    hover = HoverTool(tooltips=[
        ("fold", "@hover_fold"),
        ("x,y, y_pred", "@x, @y, @y_pred")])

    if fold == 'all':
        ds.data['color'] = ds.data.data_split.map(lambda fold_name: color_palette[fold_to_nbr(fold_name)])
        ds_filtered = ds.select(in_train_set=['out_fold'])
    else:
        ds.data['color'] = ds.data.in_train_set.map(lambda in_train: 'black' if in_train == 'in_fold' else 'lightgrey')
        ds_filtered = ds.select(in_train_set=['in_fold', 'out_fold'], data_split=fold)

    return (ds_filtered
            .to(hv.Points, kdims=['x', 'y'], vdims=['color', 'hover_fold', 'y_pred'], groupby=[])
            .opts(color='color', size=5, width=800, height=550)
            .opts('Points', tools=[hover]))


def create_line(ds, fold, show_unselected):
    unique_folds = ds.data.data_split.unique()
    hover = HoverTool(tooltips=[
        ("fold", "@data_split")
    ])

    available_colors = my_fold_colors(ds)
    alpha = .05 if show_unselected else 0
    cmap = [color if this_fold == fold else hex_to_rgb(color, alpha=alpha)
            for this_fold, color in zip(unique_folds, available_colors)]

    lines = (ds
             .sort()
             .to(hv.Curve, kdims=['x'], vdims=['y_pred'], groupby=['data_split']).overlay(['data_split'])
             .opts('Curve', color=hv.Cycle(cmap), line_width=5, tools=[hover])
             .opts(show_legend=False, width=800, height=550, xlim=(0, 11), ylim=(0, 13))
             )

    return lines


def create_dist_plot(ds, fold):
    df = ds.data
    df['absolute error'] = (df.y - df.y_pred).abs()
    df_selected = df.loc[df['data_split'] == fold].copy()
    df_selected['selected'] = fold

    df_all = df.groupby(['data_split', 'in_train_set']).mean().reset_index()
    df_all['selected'] = 'all'

    ds_dist = hv.Dataset(pd.concat([df_selected, df_all]),
                         kdims=['selected', 'in_train_set'], vdims=['absolute error'])
    opts = dict(width=600, height=540)
    boxWhisker = (ds_dist
                  .to(hv.BoxWhisker, kdims=['selected', 'in_train_set'], groupby=[])
                  .opts(**opts, ylim=(0, 4))
                  )
    violin = hv.Violin(boxWhisker).opts(**opts)
    return (boxWhisker + violin).opts(tabs=True)
