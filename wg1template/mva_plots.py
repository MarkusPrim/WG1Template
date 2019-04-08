from typing import Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

from wg1template.plot_style import set_matplotlibrc_params, xlabel_pos, ylabel_pos, KITColors


def plot_roc_curve(
        y_true: np.ndarray,
        y_predict: np.ndarray,
        fig_ax_tuple: Optional[Tuple[plt.Figure, plt.Axes]] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots the ROC curve for given true and predicted classes.

    :param y_true: True labels.
    :param y_predict: Classifier output.
    :param fig_ax_tuple: Optionally the matplotlib figure and axis objects
    to be used can be provided as tuple (fig, ax) via this parameter.
    Default is None.
    :return: A tuple of matplotlib Figure and Axes
    """
    set_matplotlibrc_params()

    if fig_ax_tuple is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig, ax = fig_ax_tuple

    fpr, tpr, _ = roc_curve(y_true, y_predict)
    roc_auc = roc_auc_score(y_true, y_predict)

    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])

    ax.plot(tpr, 1 - fpr, color=KITColors().kit_blue)
    ax.text(0.03, 0.03, f"ROC Integral = {roc_auc:.3f}")

    ax.set_xlabel("Background Suppression", xlabel_pos)
    ax.set_ylabel("Signal Efficiency", ylabel_pos)

    return fig, ax
