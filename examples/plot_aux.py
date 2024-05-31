from __future__ import annotations

import numpy as np
import pandas as pd
import h5py

from puma import VarVsEff, VarVsEffPlot
from puma.hlplots import AuxResults, Tagger
from puma.utils import get_dummy_tagger_aux, logger

fname = '../salt/salt/logs/GN2ej_20240425-T043025/ckpts/epoch=010-val_loss=0.02963__test_signal.h5'
file = h5py.File(fname, "r")

# define the tagger
GN2ej = Tagger(
    name="GN2ej",
    label="GN2ej",
    colour="deepskyblue",
    reference=True,
)

jets = file["jets"][:6000]
tracks_from_jet = file["tracks_from_jet"][:6000]

GN2ej.labels = np.array(
    jets["isDisplaced"],
    dtype=[("isDisplaced","i4")],
)
GN2ej.aux_scores = {
    "vertexing": tracks_from_jet["VertexIndex"],
    #"track_origin": tracks_from_jet["displaced"],
}
GN2ej.aux_labels = {
    "vertexing": tracks_from_jet["truthVertexIndex"],
    "track_origin": tracks_from_jet["truthOriginLabel"],
}
GN2ej.perf_vars = {"pt": jets["pt"]}

VSI = Tagger(
    name="VSI",
    label="VSI",
    colour="pink",
)
VSI.labels = np.array(
    jets["isDisplaced"],
    dtype=[("isDisplaced","i4")],
)
VSI.aux_scores = {
    "vertexing": tracks_from_jet["VSIVertexIndex"],
}
VSI.aux_labels = {
    "vertexing": tracks_from_jet["truthVertexIndex"],
    "track_origin": tracks_from_jet["truthOriginLabel"],
}
VSI.perf_vars = {"pt": jets["pt"]}

# create the AuxResults object
aux_results = AuxResults(sample="GN2ej")
aux_results.add(GN2ej)
aux_results.add(VSI)

com = "13.6"
mc = "mc21a"
sample_str = "Emerging jets"
cut_str = "$p_T$ > 200 GeV, $|\\eta| < 2.5$"

aux_results.atlas_second_tag = (
	"$\\sqrt{s}=" + com + "$ TeV, " + mc + "\n" + sample_str + ", " + cut_str
)

aux_results.plot_var_vtx_perf(vtx_flavours=["dispjet"], no_vtx_flavours=["dispjet"])

