from wg1template.histogram_plots import *
from wg1template.plot_style import TangoColors
from wg1template.plot_utilities import export


def example_simple_histogram_plot():
    hp1 = SimpleHistogramPlot(dummy_var)
    hp1.add_component("Something", data.DummyVariable, color=TangoColors.scarlet_red)
    hp2 = SimpleHistogramPlot(dummy_var)
    hp2.add_component("Else", bkg.DummyVariable, color=TangoColors.aluminium)
    fig, ax = create_solo_figure()
    hp1.plot_on(ax, ylabel="Events")
    hp2.plot_on(ax)
    add_descriptions_to_plot(
        ax,
        experiment='Belle',
        luminosity=r"$\int \mathcal{L} \,dt=711\,\mathrm{fb}^{-1}$",
        additional_info='WG1 Preliminary Plot Style\nSimpleHistogramPlot'
    )
    plt.show()
    export(fig, 'simple', 'examples')
    plt.close()


def example_stacked_histogram_plot():
    hp = StackedHistogramPlot(dummy_var)
    hp.add_component("Continum", cont.DummyVariable, weights=cont.__weight__, color=TangoColors.slate,
                     comp_type='stacked')
    hp.add_component("Background", bkg.DummyVariable, weights=bkg.__weight__, color=TangoColors.sky_blue,
                     comp_type='stacked')
    hp.add_component("Signal", sig.DummyVariable, weights=sig.__weight__, color=TangoColors.orange, comp_type='stacked')
    fig, ax = create_solo_figure()
    hp.plot_on(ax, ylabel="Candidates")
    add_descriptions_to_plot(
        ax,
        experiment='Belle',
        luminosity=r"$\int \mathcal{L} \,dt=711\,\mathrm{fb}^{-1}$",
        additional_info='WG1 Preliminary Plot Style\nStackedHistogramPlot'
    )
    plt.show()
    export(fig, 'stacked', 'examples')
    plt.close()


def example_data_mc_histogram_plot():
    hp = DataMCHistogramPlot(dummy_var)
    hp.add_mc_component("Continum", cont.DummyVariable, weights=cont.__weight__, color=TangoColors.slate)
    hp.add_mc_component("Background", bkg.DummyVariable, weights=bkg.__weight__, color=TangoColors.sky_blue)
    hp.add_mc_component("Signal", sig.DummyVariable, weights=sig.__weight__, color=TangoColors.orange)
    hp.add_data_component("Data", data)
    fig, ax = create_hist_ratio_figure()
    hp.plot_on(ax[0], ax[1], style="stacked", ylabel="Candidates")
    add_descriptions_to_plot(
        ax[0],
        experiment='Belle',
        luminosity=r"$\int \mathcal{L} \,dt=711\,\mathrm{fb}^{-1}$",
        additional_info='WG1 Preliminary Plot Style\nDataMCHistogramPlot'
    )
    plt.show()
    export(fig, 'data-mc', 'examples')
    plt.close()


def example_combo_plot():
    hp1 = StackedHistogramPlot(dummy_var)
    hp1.add_component("Continum", cont.DummyVariable, weights=cont.__weight__, color=TangoColors.slate,
                      comp_type='stacked')
    hp1.add_component("Background", bkg.DummyVariable, weights=bkg.__weight__, color=TangoColors.sky_blue,
                      comp_type='stacked')
    hp1.add_component("Signal", sig.DummyVariable, weights=sig.__weight__, color=TangoColors.orange,
                      comp_type='stacked')

    hp2 = SimpleHistogramPlot(dummy_var)
    hp2.add_component("Signal Shape x0.5", sig.DummyVariable, weights=sig.__weight__ * 0.5,
                      color=TangoColors.scarlet_red, ls='-.')

    fig, ax = create_solo_figure()
    hp1.plot_on(ax, ylabel="Candidates")
    hp2.plot_on(ax, hide_labels=True)  # Hide labels to prevent overrides)
    add_descriptions_to_plot(
        ax,
        experiment='Belle',
        luminosity=r"$\int \mathcal{L} \,dt=711\,\mathrm{fb}^{-1}$",
        additional_info='WG1 Preliminary Plot Style\nStackedHistogramPlot\n+SimpleHistogramPlot'
    )
    plt.show()
    export(fig, 'combo', 'examples')
    plt.close()


def example2_combo_plot():
    hp1 = DataMCHistogramPlot(dummy_var)
    hp1.add_mc_component("Continum", cont.DummyVariable, weights=cont.__weight__, color=TangoColors.slate)
    hp1.add_mc_component("Background", bkg.DummyVariable, weights=bkg.__weight__, color=TangoColors.sky_blue)
    hp1.add_mc_component("Signal", sig.DummyVariable, weights=sig.__weight__, color=TangoColors.orange)
    hp1.add_data_component("Data", data)

    hp2 = SimpleHistogramPlot(dummy_var)
    hp2.add_component("Signal Shape x0.5", sig.DummyVariable, weights=sig.__weight__ * 0.5,
                      color=TangoColors.scarlet_red, ls='dotted')

    fig, ax = create_hist_ratio_figure()
    hp1.plot_on(ax[0], ax[1], style='stacked', ylabel="Candidates")
    hp2.plot_on(ax[0], hide_labels=True)  # Hide labels to prevent overrides
    add_descriptions_to_plot(
        ax[0],
        experiment='Belle',
        luminosity=r"$\int \mathcal{L} \,dt=711\,\mathrm{fb}^{-1}$",
        additional_info='WG1 Preliminary Plot Style\nDataMCHistogramPlot\n+SimpleHistogramPlot'
    )
    plt.show()
    export(fig, 'combo2', 'examples')
    plt.close()


if __name__ == '__main__':
    # MC samples
    sig = pd.DataFrame({
        'DummyVariable': 5 + np.random.randn(2000) * 0.5,
        '__weight__': 0.5 * np.ones(2000)
    })
    bkg = pd.DataFrame({
        'DummyVariable': 7 + np.random.randn(2000) * 2,
        '__weight__': 0.5 * np.ones(2000),
    })
    cont = pd.DataFrame({
        'DummyVariable': np.random.uniform(0, 10, 600),
        '__weight__': 0.5 * np.ones(600)
    })

    # Generate a fake data distribution
    data = pd.concat((
        pd.DataFrame({'DummyVariable': 5 + np.random.randn(1000) * 0.5}),
        pd.DataFrame({'DummyVariable': 7 + np.random.randn(1000) * 2}),
        pd.DataFrame({'DummyVariable': np.random.uniform(0, 10, 300)}),
    ))

    # Define how this variable should be presented, bins, scope, unit label, etc...
    dummy_var = HistVariable("DummyVariable",
                             n_bins=25,
                             scope=(-0, 10),
                             var_name="DummyVariable",
                             unit="GeV")

    example_simple_histogram_plot()
    example_stacked_histogram_plot()
    example_data_mc_histogram_plot()

    example_combo_plot()
    example2_combo_plot()
