"""
This module contains utility functions for the plots in plots.py
"""

import os
from typing import Union, Optional, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def export(
        fig: plt.Figure,
        filename: Union[str, os.fspath],
        target_dir: str = "plots/",
        file_formats: Tuple[str] = (".pdf", ".png")
) -> None:
    """
    Convenience function for saving a matplotlib figure.

    :param fig: A matplotlib figure.
    :param filename: Filename of the plot without .pdf suffix.
    :param file_formats: Tuple of file formats specifying the format
    figure will be saved as.
    :param target_dir: Directory where the plot will be saved in.
    Default is './plots/'.
    :return: None
    """
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    for file_format in file_formats:
        fig.savefig(os.path.join(target_dir, f'{filename}{file_format}'), bbox_inches="tight")


# TODO maybe this should be split into four different functions???
def add_cut_to_axis(ax: plt.axis,
                    cut_left: Optional[float] = None,
                    cut_right: Optional[float] = None,
                    cut_window: Optional[Tuple[float, float]] = None,
                    keep_window: Optional[Tuple[float, float]] = None,
                    color: str = 'white'):
    """
    Adds a "cut" to a given axis. The cut is shown as shaded area with the color
    given in the parameter color.

    :param ax: Axis to plot on.
    :param cut_left: Upper x value of the cut. If set, the area with
    x < cut_left is indicated to be cut away. Default is None
    :param cut_right: Lower x value of the cut. If set, the area with
    x > cut_right is indicated to be cut away. Default is None
    :param cut_window:
    :param keep_window:
    :param color: Color of the overlay of the area which is indicated to be
    cut away. Default is 'white'.
    """
    x_lim_low, x_lim_high = ax.get_xlim()

    if cut_left is not None:
        ax.axvspan(x_lim_low, cut_left, facecolor=color, alpha=0.7)
        ax.axvline(cut_left, color='black', linestyle='dashed', lw=1.5, label='Cut')
    elif cut_right is not None:
        ax.axvspan(cut_right, x_lim_high, facecolor=color, alpha=0.7)
        ax.axvline(cut_right, color='black', linestyle='dashed', lw=1.5, label='Cut')
    elif cut_window is not None:
        ax.axvspan(cut_window[0], cut_window[1], facecolor=color, alpha=0.7)
        ax.axvline(cut_window[0], color='black', linestyle='dashed', lw=1.5, label='Cut')
        ax.axvline(cut_window[1], color='black', linestyle='dashed', lw=1.5)
    elif keep_window is not None:
        ax.axvspan(x_lim_low, keep_window[0], facecolor=color, alpha=0.7)
        ax.axvline(keep_window[0], color='black', linestyle='dashed', lw=1.5, label='Cut')
        ax.axvspan(keep_window[1], x_lim_high, facecolor=color, alpha=0.7)
        ax.axvline(keep_window[1], color='black', linestyle='dashed', lw=1.5, label='Cut')

    ax.legend(frameon=False, bbox_to_anchor=(1, 1))


# TODO maybe move the next two function to a more appropriate module
def get_auto_binning(
        df: pd.DataFrame, variable: str,
        number_of_bins: int = 100
) -> Tuple[int, Union[int, float], Union[int, float]]:
    """
    Calculates the binning for the given column in the pandas data frame.

    :param df: Pandas data frame.
    :param variable: Name of the column (variable) to be binned.
    :param number_of_bins: initial value for binning algorithm
    :return: Binning as tuple of (number_of_bins, min_val, max_val)
    """
    # TODO: replace hardcoded values with something more dynamic
    # TODO: Binning only between fixed min and max values
    # TODO: Centered binning

    min_val = df[variable].min()
    max_val = df[variable].max()

    if np.issubdtype(df[variable].dtype, np.integer):
        max_val += 1
        if max_val - min_val < number_of_bins:
            number_of_bins = int(max_val - min_val)
    else:
        # TODO: Fix the case for float values:
        if max_val - min_val >= (number_of_bins / 2):
            number_of_bins = int(max_val - min_val)

    if number_of_bins > 200:
        number_of_bins = 100

    return number_of_bins, min_val, max_val


def get_auto_binning_for_compound_df(
        dfs: Dict[str, pd.DataFrame],
        variable: str,
        number_of_bins: int = 0,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
) -> Tuple[int, Union[int, float], Union[int, float]]:
    """
    Calculates the binning for a column in a dictionary of pandas data frames.

    :param dfs: Dictionary of pandas data frames.
    :param variable: Name of the column (variable) to be binned.
    :param number_of_bins: initial value for binning algorithm
    :param min_val: initial value for binning algorithm
    :param max_val: initial value for the binning algorithm
    :return: Binning as tuple of (number_of_bins, min_val, max_val)
    """

    for df in dfs.values():
        tag_binning = get_auto_binning(df, variable)
        if tag_binning[0] > number_of_bins:
            number_of_bins = tag_binning[0]
        if min_val is None:
            min_val = tag_binning[1]
        elif tag_binning[1] < min_val:
            min_val = tag_binning[1]
        if max_val is None:
            max_val = tag_binning[2]
        elif tag_binning[2] > max_val:
            max_val = tag_binning[2]

    return number_of_bins, min_val, max_val
