"""Produce histogram of discriminant from tagger output and labels."""

import numpy as np
from ftag import Flavours

from puma import Roc, RocPlot
from puma import Histogram, HistogramPlot
from puma.metrics import calc_rej
from puma.utils import get_dummy_2_taggers, logger, get_good_linestyles
import h5py
import pandas as pd


#file_path = '../logs/GN2ej_20231021-T185330/ckpts/epoch=017-val_loss=0.02767__test_pp_output_train.h5'
file_paths = [
		'../salt/salt/logs/GN2ej_20240423-T063340/ckpts/epoch=015-val_loss=0.02852__test_pp_output_val.h5',
		'../salt/salt/logs/GN2ej_20240425-T043025/ckpts/epoch=010-val_loss=0.02963__test_pp_output_val.h5'
]

labels = [
    "GN2ej",
    "GN2ej_noPt"
]

plot_roc = RocPlot(
    n_ratio_panels=0,
    ylabel="QCD rejection",
    xlabel="EJ efficiency",
    atlas_second_tag=f"$\\sqrt{{s}}=13.6$ TeV",
    figsize=(6.5, 6),
    y_scale=1.4,
)

for file_path, label in zip(file_paths,labels):
    with h5py.File(file_path, 'r') as hdf_file:

        ds = hdf_file['jets']

        df = pd.DataFrame({
            'isDisplaced': np.array(ds['isDisplaced']).transpose(), 
            'GN2ej_pdispjet': np.array(ds['GN2ej_pdispjet']).transpose(),
        })
        df = df.dropna()

        # defining boolean arrays to select the different flavour classes
        is_pu = df["isDisplaced"] == 0
        is_hs = df["isDisplaced"] == 1

        # defining target efficiency
        sig_eff = np.linspace(0.75, 1, 100)

        n_pu = sum(is_pu)

        rej = calc_rej(df[is_hs]["GN2ej_pdispjet"].values, df[is_pu]["GN2ej_pdispjet"].values, sig_eff)

        # here the plotting of the roc starts
        plot_roc.add_roc(
            Roc(
                sig_eff,
                rej,
                n_test=n_pu,
                rej_class="ujets",
                signal_class="bjets",
                label=label,
            ),
            reference=False,
        )

plot_roc.draw()
plot_roc.savefig(f"roc.png", transparent=False)

