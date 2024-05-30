from puma import Roc, RocPlot
from puma.metrics import calc_rej
from plotter.config_dict import ConfigDict
import h5py
import numpy as np
import pandas as pd
from plotter.plot_classes.plotbase import PlotBase


class RocPlotBase(PlotBase):
    def plot(self):
        required_params = {
            "n_ratio_panels",
            "ylabel",
            "xlabel",
            "atlas_second_tag",
            "figsize",
            "y_scale",
        }
        filtered_params = {
            key: value for key, value in self.config.style.items() if key in required_params
        }
        roc_plot = RocPlot(**filtered_params)

        for _, sample in self.config.samples.items():
            sample_config = ConfigDict(sample)
            with h5py.File(sample_config.path, "r") as hdf_file:
                ds = hdf_file[sample_config.df_name]

                target_label = self.config.target_label
                tagger_output = self.config.tagger_output

                df = pd.DataFrame(
                    {
                        target_label: np.array(ds[target_label]).transpose(),
                        tagger_output: np.array(ds[tagger_output]).transpose(),
                    }
                )
                df = df.dropna()

                # defining boolean arrays to select the different flavour classes
                is_pu = df[target_label] == 0
                is_hs = df[target_label] == 1

                # defining target efficiency
                sig_eff = np.linspace(*self.config.range)

                n_pu = sum(is_pu)

                rej = calc_rej(
                    df[is_hs]["GN2ej_pdispjet"].values, df[is_pu]["GN2ej_pdispjet"].values, sig_eff
                )

                # here the plotting of the roc starts
                roc_plot.add_roc(
                    Roc(
                        sig_eff,
                        rej,
                        n_test=n_pu,
                        rej_class="qcd",
                        label=sample_config.label,
                    ),
                    reference=sample_config.reference,
                )
                roc_plot.set_ratio_class(1, "qcd")

        roc_plot.draw()
        roc_plot.savefig(self.config.file_name, transparent=False)
