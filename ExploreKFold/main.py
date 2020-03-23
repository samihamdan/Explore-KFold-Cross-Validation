import param
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

import panel as pn
import holoviews as hv

from my_figures import create_scatter, create_line, create_dist_plot
from computations import data_generator, KFold_split, fit_transform
from settings import MAX_N, MAX_N_FOLDS, N_EXTERNAL_TEST

hv.extension('bokeh')


def PolyRegression(degree): return make_pipeline(PolynomialFeatures(degree), LinearRegression())


class Dashboard(param.Parameterized):
    # Setup
    model_options = {'Linear Regression': LinearRegression(), **{f'Poly Degree {i}': PolyRegression(i)
                                                                 for i in range(2, 16, 2)}}
    relationsships = {'linear': 'linear', 'sine wave': 'sine_wave'}

    # Widgets for controling simulation
    n = param.Integer(default=100, bounds=(20, MAX_N), step=20)
    Noise_Amplitude = param.Number(default=1, bounds=(0, 10))
    noise = param.ObjectSelector(default='normal', objects=['normal', 'constant'])
    Underlying_Relation = param.ObjectSelector(default=relationsships['linear'], objects=relationsships)

    # Widgets for modeling
    estimator = param.ObjectSelector(default=model_options['Linear Regression'], objects=model_options)
    N_Folds = param.Integer(default=10, bounds=(5, MAX_N_FOLDS))
    Shuffle_Folds = param.Boolean(False)

    # Widgets for changing visuals
    Show_Unselected_Folds = param.Boolean(True)
    Select_Fold = param.ObjectSelector(default='all', objects={'all': 'all', **{f'fold:{fold}': f'fold:{fold}'
                                                                                for fold in range(N_Folds.default)}}
                                       )
    # interactive changes on data
    data = param.DataFrame(data_generator(n.default, noise.default,
                                          Noise_Amplitude.default, Underlying_Relation.default), precedence=-1)
    data_extra = param.DataFrame(data_generator(N_EXTERNAL_TEST, noise.default,
                                                Noise_Amplitude.default, Underlying_Relation.default), precedence=-1)
    data_splitted = param.DataFrame(KFold_split(data.default, data_extra.default, N_Folds.default, False),
                                    precedence=-1)
    data_plot = param.DataFrame(fit_transform(data_splitted.default, estimator.default), precedence=-1)

    @param.depends('n', 'noise', 'Noise_Amplitude', 'Underlying_Relation', watch=True)
    def update_data_creation(self):
        self.data = data_generator(self.n, self.noise,
                                   self.Noise_Amplitude, self.Underlying_Relation)
        self.data_extra = data_generator(N_EXTERNAL_TEST, self.noise,
                                         self.Noise_Amplitude, self.Underlying_Relation)

    @param.depends('data', 'data_extra', 'N_Folds', 'Shuffle_Folds', watch=True)
    def update_split(self):

        # So that never try more folds than samples => would run into an error in sklearn
        if self.N_Folds > self.n:
            self.N_Folds = self.n
        if self.N_Folds > self.param['N_Folds'].bounds[1]:
            self.N_Folds = self.param['N_Folds'].bounds[1]

        self.param['N_Folds'].bounds = (5, min(MAX_N_FOLDS, self.n))

        self.data_splitted = KFold_split(self.data, self.data_extra, self.N_Folds,
                                         self.Shuffle_Folds)

    @param.depends('data_splitted', 'estimator', watch=True)
    def update_estimator(self):
        self.data_plot = fit_transform(self.data_splitted, self.estimator)

    @param.depends('data_plot', 'Show_Unselected_Folds', 'Select_Fold')
    def view(self):
        ds = hv.Dataset(self.data_plot, kdims=['x', 'data_split', 'in_train_set', 'y'], vdims=['y_pred'])

        self.param['Select_Fold'].objects = {'all': 'all', **{f'fold:{fold}': f'fold:{fold}'
                                                              for fold in range(self.N_Folds)}}

        scatter = ds.apply(create_scatter, fold=self.Select_Fold)
        lines = ds.apply(create_line, fold=self.Select_Fold, show_unselected=self.Show_Unselected_Folds)
        dist_plot = ds.apply(create_dist_plot, fold=self.Select_Fold)
        return pn.Column(pn.Row((scatter * lines), dist_plot))


dashboard = Dashboard(name='')

simulation_widgets = pn.Column(pn.pane.Markdown('## Simulation Options', height=5),
                               pn.panel(dashboard.param,
                                        parameters=['n', 'Underlying_Relation', 'noise', 'Noise_Amplitude']),
                               css_classes=['widget-box'])

modeling_widgets = pn.Column(pn.pane.Markdown('## Modeling Options', height=5),
                             pn.panel(dashboard.param, parameters=['N_Folds', 'Shuffle_Folds', 'estimator']),
                             css_classes=['widget-box'])

viewing_widgets = pn.Row(pn.panel(dashboard.param, parameters=['Select_Fold'],
                                  sizing_mode='stretch_width'),
                         pn.panel(dashboard.param, parameters=['Show_Unselected_Folds'], margin=(30, 0)))

dash = pn.Column(pn.pane.Markdown('# Exploring KFold Cross-Validation', align='center'),
                 pn.Row(pn.Column(dashboard.view, viewing_widgets),
                        pn.Column(simulation_widgets, pn.Spacer(heigth=150), modeling_widgets, margin=(40, 10))))

doc = dash.server_doc()
doc.title = 'Explore Cross-Validation'
