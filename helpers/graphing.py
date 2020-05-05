import plotly.express as px
import plotly.io as pio
import pydotplus

from IPython.display import Image
from sklearn.calibration import calibration_curve
from sklearn.externals.six import StringIO
from sklearn.tree import DecisionTreeClassifier, export_graphviz


def draw_tree_splits(model, pngname='dtree.png'):
    """A way to graph how simple tree splits on a feature
    Helps visualize how decision is made, could be used to investigate and
    come up with more linear features
    """
    dot_data = StringIO()
    export_graphviz(model, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(pngname)
    return Image(graph.create_png())


def calibration_graph(actual, pred_prob, name, n_bins=20):
    """Draw calibration curve with plotly io"""
    fraction_of_positives, mean_predicted_value = calibration_curve(actual, pred_prob, n_bins=n_bins)
    bins = [val * 1.0/n_bins for val in range(0, n_bins + 1)]
    fig = {
        "data": [{"type": "scatter",
                  "x": mean_predicted_value,
                  "y": fraction_of_positives,
                  "name": name, 'mode': 'lines+markers'},
                 {"type": "scatter", "x": bins, "y": bins, "name": 'perfectly calibrated',
                  "line": {'color': 'red', 'dash': 'dot'}}],
        "layout": {"title": {"text": f"Calibration curve for {name}"},
                   "width": 800, "height": 600,
                   "xaxis": {'title': 'mean predicted value'},
                   "yaxis": {'title': 'fraction of positives'}}
    }
    pio.show(fig)


def draw_roc_curve(test_real, test_pred, test_pred_proba):
    """This needs update with new plotly express instead"""
    logit_roc_auc = roc_auc_score(test_real, test_pred)
    fpr, tpr, thresholds = roc_curve(test_real, test_pred_proba)

    fig_data = [{'x': fpr, 'y': tpr, 'name': 'ROC (AUC = %0.2f)' % logit_roc_auc},
                {'x': [0,1], 'y': [0,1], 'line': {'color': 'red', 'dash': 'dash'}, 'name': 'x=y'}]
    fig_layout = {'title': 'Receiver operating characteristic',
                  'xaxis': {'title': 'False Positive Rate'},
                  'yaxis': {'title': 'True Positive Rate'},
                 }
    fig = {'data': fig_data, 'layout': fig_layout}
    py.iplot(fig)


def train_dtree(train, features, test, max_depth=5, return_model=False):
    dtc = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=1000, random_state=34, max_depth=max_depth)
    dtc.fit(train[features], train.booked)
    y_pred = dtc.predict(test[features])
    y_pred_class = [int(label >= .5) for label in y_pred]

    scores = training_scores(test, y_pred, y_pred_class)
    if return_model:
        return dtc, scores
    return scores
