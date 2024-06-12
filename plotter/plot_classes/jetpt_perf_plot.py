from puma import VarVsEff, VarVsEffPlot
from plotter.config_dict import ConfigDict
import h5py
import numpy as np
import pandas as pd
from plotter.plot_classes.plotbase import PlotBase

class JetPtPerfPlotBase(PlotBase):
	"""
	Subclass of PlotBase to plot signal and background efficiency performance metrics vs jet p_T.
	"""

	def plot(self):
		# INITIALIZING figure plot base
		# -----------------------------
		# required parameters for plot figure
		required_params = {
			"grid",
			"figsize",
			"atlas_second_tag",
			"y_scale",
			"atlas_tag_outside",
		}
		# filtering out necessary parameters from config file
		filtered_params = {
			key: value for key, value in self.config.style.items() if key in required_params
		}

		# define the plots
		plot_sig_eff = VarVsEffPlot(
			mode="sig_eff",
			ylabel="Emerging jet efficiency",
			xlabel=r"$p_{T}$ [GeV]",
			logy=False,
			**filtered_params
		)

		plot_bkg_rej = VarVsEffPlot(
			mode="bkg_rej",
			ylabel="QCD jet rejection",
			xlabel=r"$p_{T}$ [GeV]",
			logy=False,
			**filtered_params
		)


		# OPEN OUTPUT DATA FILE AND PROCESS IT
		# ------------------------------------
		for _, sample in self.config.samples.items():
			print(sample)
			sample_config = ConfigDict(sample)
			with h5py.File(sample_config.path, "r") as hdf_file:
				ds_jet = hdf_file["jets"]

				keys_list = list(ds_jet.dtype.fields.keys())

				# get the working point
				wp = self.config.working_point

				# string names for probability of displaced and prompt
				pDisp = keys_list[-2]
				pPrompt = keys_list[-1]

				# extract pDisp, pPrompt, and jet p_T, store in pandas dataframe
				df = pd.DataFrame(
					{
						"pt": np.array(ds_jet["pt"])/1000, # jet p_T in GeV
						"isDisplaced": np.array(ds_jet["isDisplaced"]),
						pDisp: np.array(ds_jet[pDisp]),
						pPrompt: np.array(ds_jet[pPrompt])
					}
				)	

				# obtain GNN discriminant values
				discs_gnn = df[pDisp]

				# define boolean arrays to select the different flavour classes
				is_disp = df["isDisplaced"] == 1
				is_prompt = df["isDisplaced"] == 0

				pt = df["pt"].values	# jet p_T in GeV


				# CREATE BINS
				# -----------
				factor = self.config.factor # factor by which each bin increases in size

				lowest_pT = np.min(df['pt'].values)
				highest_pT = np.max(df[is_disp]['pt'].values)

				# create bin edges array and keep track of current edge
				binedges = [lowest_pT, lowest_pT+self.config.starting_bin_width]
				current_edge = binedges[-1]

				# calculate all bin edges
				j = 0
				while current_edge < highest_pT:
					current_edge += (binedges[j+1]-binedges[j])*(factor)
					binedges.append(current_edge)
					j += 1

				print(binedges)


				# DEFINE THE CURVES
				# -----------------
				gnn_ej = VarVsEff(
					x_var_sig = pt[is_disp],
					disc_sig = discs_gnn[is_disp].values,
					x_var_bkg = pt[is_prompt],
					disc_bkg = discs_gnn[is_prompt].values,
					bins = binedges,
					working_point = None,
					disc_cut = wp,
					label = sample_config.label,
					colour = sample_config.colour,
					linewidth = 1.2
				)


				# ADD THE CURVES TO THE PLOTS
				# ---------------------------
				plot_sig_eff.add(gnn_ej, reference=True)
				plot_sig_eff.atlas_second_tag += f", Score > {wp}"

				plot_bkg_rej.add(gnn_ej, reference=True)
				plot_bkg_rej.atlas_second_tag += f", Score > {wp}"


		# DRAW AND SAVE THE PLOTS
		# -----------------------
		plot_sig_eff.draw()
		plot_sig_eff.savefig("EJ_eff_vs_jetpT.png", transparent=False)

		plot_bkg_rej.draw()
		plot_bkg_rej.savefig("QCD_rej_vs_jetpT.png", transparent=False)
