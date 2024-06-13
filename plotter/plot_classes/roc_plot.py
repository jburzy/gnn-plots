from puma import Roc, RocPlot
from puma.metrics import calc_eff, calc_rej
from plotter.config_dict import ConfigDict
import h5py
import numpy as np
import matplotlib.pyplot as plt
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

                # get attribute name for GNN ej score
                keys_list = list(ds.dtype.fields.keys())
                pDisp = keys_list[-2]

                df = pd.DataFrame(
                    {
                        target_label: np.array(ds[target_label]).transpose(),
                        pDisp: np.array(ds[pDisp]).transpose(),
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
                    df[is_hs][pDisp].values, df[is_pu][pDisp].values, sig_eff
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

                # add cut values and background rejections if desired
                if self.config.show_cuts == True:
                    # calculate the efficiencies for the specific cut values
                    if self.config.cut_values is not None:
                        cut_values = self.config.cut_values

                    sig_disc = df[is_hs][pDisp]     # convenient to store signal discriminants
                    N_signal = len(sig_disc)

                    cut_effs = []   # initialize array to store efficiencies from cut values
                    cut_rejs = []   # initialize array to store corresponding bkg effs from cuts

                    for cut in cut_values:
                        true_pos = sig_disc[sig_disc >= cut]    # determine the signal that passes the cut
                        eff_ = len(true_pos)/N_signal
                        rej_ = calc_rej(
                            df[is_hs][pDisp].values, df[is_pu][pDisp].values, eff_
                        )
                        cut_effs.append(eff_)
                        cut_rejs.append(rej_)

                    # plot the cuts and the corresponding bkg rejs
                    markers = ['o', 'v', 's', 'P', 'X', 'd']
                    for i in range(len(cut_values)):
                        roc_plot.axis_top.scatter(
                            cut_effs[i], 
                            cut_rejs[i], 
                            s = 75,
                            marker = markers[i],
                            facecolors = list(roc_plot.label_colours.values())[-1],
                            edgecolors = 'black',
                            alpha=0.6,
                            label = "cut = {0:.3f}, rej. = {1:.2e}".format(cut_values[i], cut_rejs[i]),
                            zorder = 99
                        )

        roc_plot.draw()

        # remove the previous legend
        roc_plot.axis_top.get_legend().remove()

        # update the legend
        if self.config.show_cuts == True:
            roc_plot.axis_top.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1,1))
        else:
            roc_plot.axis_top.legend()


        roc_plot.savefig(self.config.file_name, transparent=False)
