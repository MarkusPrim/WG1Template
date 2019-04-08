from wg1template.histogram_plots import *
from wg1template.plot_style import TangoColors

# MC samples
sig = pd.DataFrame({
    'DummyVariable': 5 + np.random.randn(20000) * 0.5,
    '__weight__': 0.5 * np.ones(20000)
})
bkg = pd.DataFrame({
    'DummyVariable': 7 + np.random.randn(20000) * 2,
    '__weight__': 0.5 * np.ones(20000),
})
cont = pd.DataFrame({
    'DummyVariable': np.random.uniform(0, 10, 6000),
    '__weight__': 0.5 * np.ones(6000)
})

# Generate a fake data distribution
datsig = pd.DataFrame({'DummyVariable': 5 + np.random.randn(10000) * 0.5})
datbkg = pd.DataFrame({'DummyVariable': 7 + np.random.randn(10000) * 2})
datcont = pd.DataFrame({'DummyVariable': np.random.uniform(0, 10, 3000)})
data = pd.concat((datsig, datbkg, datcont))

dummy_var = HistVariable("DummyVariable",
                         n_bins=25,
                         scope=(-0, 10),
                         var_name="DummyVariable",
                         unit="GeV")

hp = DataMCHistogramPlot(dummy_var)

hp.add_mc_component("Continum", cont.DummyVariable, weights=cont.__weight__, color=TangoColors.slate)
hp.add_mc_component("Background", bkg.DummyVariable, weights=bkg.__weight__, color=TangoColors.sky_blue)
hp.add_mc_component("Signal", sig.DummyVariable, weights=sig.__weight__, color=TangoColors.orange)

hp.add_data_component("Data", data)

fig, ax = create_hist_ratio_figure()
hp.plot_on(ax[0], ax[1], "stacked", "Candidates")
plt.show()
