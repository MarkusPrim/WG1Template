from typing import Union

import matplotlib.pyplot as plt
import numpy as np

import wg1template.plot_style as plot_style

plot_style.set_matplotlibrc_params()


class DataVariable:

    def __init__(self,
                 x_name: Union[str, None] = None,
                 x_unit: Union[str, None] = None,
                 y_name: Union[str, None] = None,
                 y_unit: Union[str, None] = None,
                 ):
        self.var_name = x_name
        self.x_unit = x_unit
        self.x_label = x_name + f' in {x_unit}' if x_unit else x_name
        self.y_unit = y_unit
        self.y_label = y_name + f' in {y_unit}' if y_unit else y_name


class DataPoints:

    def __init__(self,
                 x_values: Union[list, np.array],
                 y_values: Union[list, np.array],
                 x_errors: Union[list, np.array, None] = None,
                 y_errors: Union[list, np.array, None] = None,
                 ):
        self.x_values = x_values
        self.y_values = y_values
        self.x_errors = x_errors
        self.y_errors = y_errors


class DataComponent:

    def __init__(self,
                 label: str,
                 data: DataPoints,
                 color: Union[str, None] = 'black',
                 ls: str = '',
                 marker: str = 'o'):

        self.label = label
        self.data = data
        self.color = color
        self.ls = ls
        self.marker = marker


class DataPointsPlot:

    def __init__(self, data_variable: DataVariable):
        self.variable = data_variable
        self.labels = []
        self.components = []

    def add_component(self,
                      label: str,
                      data_points: DataPoints,
                      color: Union[str, None] = 'black',
                      ls: str = '',
                      marker: str = 'o',
                      ):
        self.components.append(DataComponent(
            label,
            data_points,
            color,
            ls,
            marker
        ))

    def plot_on(self,
                ax: plt.axis,
                draw_legend: bool = True,
                legend_inside: bool = True,
                legend_kwargs: dict = {},
                yaxis_scale=1.3,
                hide_labels: bool = False) -> plt.axis:

        for component in self.components:
            (_, caps, _) = ax.errorbar(
                component.data.x_values,
                component.data.y_values,
                yerr=component.data.y_errors,
                xerr=component.data.x_errors,
                label=component.label,
                color=component.color,
                ls=component.ls,
                marker=component.marker,
            )

        if not hide_labels:
            ax.set_xlabel(self.variable.x_label, plot_style.xlabel_pos)
            ax.set_ylabel(self.variable.y_label, plot_style.ylabel_pos)

        if draw_legend:
            if legend_inside:
                ax.legend(frameon=False, **legend_kwargs)
                ylims = ax.get_ylim()
                ax.set_ylim(ylims[0], yaxis_scale * ylims[1])
            else:
                ax.legend(frameon=False, bbox_to_anchor=(1, 1), **legend_kwargs)

        return ax
