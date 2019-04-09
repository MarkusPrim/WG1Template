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
        self.y_label = y_name + f' in {y_unit}' if y_unit else x_name


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
                 ls: str = ''):

        self.label = label
        self.data = data
        self.color = color
        self.ls = ls


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
                      ):
        self.components.append(DataComponent(
            label,
            data_points,
            color,
            ls,
        ))

    def plot_on(self,
                ax: plt.axis,
                draw_legend: bool = True,
                legend_inside: bool = True,
                yaxis_scale=1.3,
                hide_labels: bool = False) -> plt.axis:

        for component in self.components:
            ax.errorbar(
                component.data.x_values,
                component.data.y_values,
                yerr=component.data.y_errors,
                xerr=component.data.x_errors,
                label=component.label,
                color=component.color,
                ls=component.ls,
            )
            # ax.hist(x=component.data,
            #         bins=bin_edges,
            #         density=normed,
            #         weights=component.weights,
            #         histtype=component.histtype,
            #         label=component.label,
            #         edgecolor=edge_color if edge_color is not None else component.color,
            #         alpha=alpha,
            #         lw=1.5,
            #         ls=component.ls,
            #         color=component.color)

        if not hide_labels:
            ax.set_xlabel(self.variable.x_label, plot_style.xlabel_pos)
            ax.set_ylabel(self.variable.y_label, plot_style.ylabel_pos)

        if draw_legend:
            if legend_inside:
                ax.legend(frameon=False)
                ylims = ax.get_ylim()
                ax.set_ylim(ylims[0], yaxis_scale * ylims[1])
            else:
                ax.legend(frameon=False, bbox_to_anchor=(1, 1))

        return ax
