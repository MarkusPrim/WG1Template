"""
This file contains general settings for plots using matplotlib.
"""
import matplotlib.pyplot as plt

from cycler import cycler

from collections import OrderedDict


class KITColors(object):
    """
    Provides the KIT colors.
    """
    kit_green = '#009682'
    kit_blue = '#4664aa'
    kit_maygreen = '#8cb63c'
    kit_yellow = '#fce500'
    kit_orange = '#df9b1b'
    kit_brown = '#a7822e'
    kit_red = '#a22223'
    kit_purple = '#a3107c'
    kit_cyan = '#23a1e0'
    kit_black = '#000000'
    light_grey = '#bdbdbd'
    grey = '#797979'
    dark_grey = '#4e4e4e'

    default_colors = [
        kit_blue,
        kit_orange,
        kit_green,
        kit_red,
        kit_purple,
        kit_brown,
        kit_yellow,
        dark_grey,
        kit_cyan,
        kit_maygreen
    ]


class TangoColors(object):
    """
    Provides the Tango colors.
    """
    scarlet_red_light = '#ef2929'
    scarlet_red = '#cc0000'
    scarlet_red_dark = '#a40000'

    aluminium_light = '#eeeeec'
    aluminium = '#d3d7cf'
    aluminium_dark = '#babdb6'

    butter_light = '#fce94f'
    butter = '#edd400'
    butter_dark = '#c4a000'

    chameleon_light = '#8ae234'
    chameleon = '#73d216'
    chameleon_dark = '#4e9a06'

    orange_light = '#fcaf3e'
    orange = '#f57900'
    orange_dark = '#ce5c00'

    chocolate_light = '#e9b96e'
    chocolate = '#c17d11'
    chocolate_dark = '#8f5902'

    sky_blue_light = '#729fcf'
    sky_blue = '#3465a4'
    sky_blue_dark = '#204a87'

    plum_light = '#ad7fa8'
    plum = '#75507b'
    plum_dark = '#5c3566'

    slate_light = '#888a85'
    slate = '#555753'
    slate_dark = '#2e3436'

    default_colors = [
        sky_blue,
        orange,
        chameleon,
        scarlet_red,
        plum,
        chocolate,
        butter,
        slate,
        aluminium,
    ]


# these variables can be given as arguments to matplotlib
# set_x/y_label functions in order to align the text to
# end of the axes
xlabel_pos = OrderedDict([('x', 1), ('ha', 'right')])
ylabel_pos = OrderedDict([('y', 1), ('ha', 'right')])

kit_color_cycler = cycler("color", KITColors.default_colors)
tango_color_cycler = cycler("color", TangoColors.default_colors)


def set_matplotlibrc_params():
    """
    Sets default parameters in the matplotlibrc.
    :return: None
    """
    xtick = {
        'top': True,
        'minor.visible': True,
        'direction': 'in',
        'labelsize': 10
    }

    ytick = {
        'right': True,
        'minor.visible': True,
        'direction': 'in',
        'labelsize': 10
    }

    axes = {
        'labelsize': 12,
        "prop_cycle": tango_color_cycler,
        'formatter.limits': (-4, 4),
        'formatter.use_mathtext': True,
        'titlesize': 'large',
        'labelpad': 4.0,
    }
    lines = {
        'lw': 1.5
    }
    legend = {
        'frameon': False
    }

    plt.rc('lines', **lines)
    plt.rc('axes', **axes)
    plt.rc('xtick', **xtick)
    plt.rc('ytick', **ytick)
    plt.rc('legend', **legend)


def set_default_colors(color_cycler):
    plt.rc('axes', prop_cycle=color_cycler)
