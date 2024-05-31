import numpy as np
import pandas as pd
import h5py

from puma import VarVsEff, VarVsEffPlot

#file_path = '../salt/salt/logs/GN2ej_20240423-T063340/ckpts/epoch=015-val_loss=0.02852__test_pp_output_val.h5'
file_path = '../salt/salt/logs/GN2ej_20240425-T043025/ckpts/epoch=010-val_loss=0.02963__test_pp_output_val.h5'

# load the jets dataset from the h5 file
with h5py.File(file_path, "r") as h5file:
    jets = pd.DataFrame(h5file["jets"][:])

# define a small function to calculate discriminant
def disc_fct(arr: np.ndarray) -> np.ndarray:
    # you can adapt this for your needs
    return arr[0] 

# calculate discriminant
discs_gnn = np.apply_along_axis(
    disc_fct, 1, jets[["GN2ej_pdispjet"]].values
)

# Getting jet pt in GeV
pt = jets["pt"].values / 1e3
# defining target efficiency
sig_eff = np.linspace(0.49, 1, 20)
# defining boolean arrays to select the different flavour classes
is_disp = jets["isDisplaced"] == 1
is_prompt = jets["isDisplaced"] == 0

# here the plotting starts

# define the curves
gnn_ej = VarVsEff(
    x_var_sig=pt[is_disp],
    disc_sig=discs_gnn[is_disp],
    x_var_bkg=pt[is_prompt],
    disc_bkg=discs_gnn[is_prompt],
    bins=[200,250,300,400,500,750,1000,3000],
    working_point=None,
    disc_cut=0.95,
    label="GN2ej",
)


# You can choose between different modes: "sig_eff", "bkg_eff", "sig_rej", "bkg_rej"
plot_sig_eff = VarVsEffPlot(
    mode="sig_eff",
    ylabel="Emerging jet efficiency",
    xlabel=r"$p_{T}$ [GeV]",
    logy=False,
    atlas_second_tag="$\\sqrt{s}=13.6$ TeV",
    figsize=(6, 4.5),
    n_ratio_panels=0,
)
plot_sig_eff.add(gnn_ej, reference=True)
plot_sig_eff.atlas_second_tag += "\nScore > 0.95"

# If you want to inverse the discriminant cut you can enable it via
# plot_sig_eff.set_inverse_cut()
plot_sig_eff.draw()
# Drawing a hline indicating inclusive efficiency
plot_sig_eff.draw_hline(0.7)
plot_sig_eff.savefig("tutorial_pt_b_eff.png", transparent=False)

# reuse the VarVsEff objects that were defined for the previous exercise
plot_bkg_rej = VarVsEffPlot(
    mode="bkg_rej",
    ylabel="QCD jet rejection",
    xlabel=r"$p_{T}$ [GeV]",
    logy=False,
    atlas_second_tag="$\\sqrt{s}=13.6$ TeV",
    figsize=(6, 4.5),
    n_ratio_panels=0,
)
plot_bkg_rej.atlas_second_tag += "\nScore > 0.95"
plot_bkg_rej.add(gnn_ej, reference=True)

plot_bkg_rej.draw()
plot_bkg_rej.savefig("tutorial_pt_light_rej.png")
