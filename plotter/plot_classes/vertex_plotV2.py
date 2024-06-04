from puma.matshow import MatshowPlot
from plotter.config_dict import ConfigDict
import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
from plotter.plot_classes.plotbase import PlotBase


def make_VImats(true_vi, pred_vi, pred_pileup, pred_fake, pred_prompt, pred_disp):
    """
    Produce both truth and prediction vertex matrices for a specific sample jet. Note that all 
    input arrays assume that non valid tracks have been removed, and arrays have been sorted and
    reorderd based on truth vertex indices.
    
    Parameters:
    ----------
        true_vi: array of truth vertex indices
        pred_vi: array of model predicted vertex indices
        pred_pileup: array of predicted pileup origins
        pred_fake: array of predicted fake origins
        pred_prompt: array of predicted prompt origins
        pred_disp: array of predicted displaced origins

    Returns:
    -------
        vi_matrices: list, contains both truth and predicted vertex index plots
    """

    # initialize list to store vertex index matrices
    vi_matrices = []

    n = len(true_vi)

    # initialize vertex index (vi) matrices as completely unpaired
    # note in these matrices: 0 -> track pairs, 1 -> tracks not paired
    mat_true = np.ones((n,n))
    mat_pred = np.ones((n,n))

    # create truth vi matrices
    for i in range(n):
        # truth vertex: checking if the i^th track is pileup (i.e. not a valid vertex)
        if true_vi[i] == -2:
            mat_true[i][i] = 0

        # constructing vertex index relationships for valid tracks
        else:
            # check for vi pairs between the i^th and j^th tracks
            for j in range(n):
                # checking for matching vertex pairs
                if true_vi[j] == true_vi[i]:
                    mat_true[i][j] = 0

    vi_matrices.append(mat_true)
    
    # create predicted vi matrix
    for i, (pu, fk, pr, dp) in enumerate(zip(pred_pileup, pred_fake, pred_prompt, pred_disp)):
        # determining predicted origin type of track i
        origin = max(pu, fk, pr, dp)

        # checking if predicted origin is pilup or fake
        if (origin == pu) or (origin == fk):
            mat_pred[i][i] = 0

        # checking if predicted origin is prompt
        elif (origin == pr) or (origin == dp):
            # check for vi pairs between the i^th and j^th tracks
            for j in range(n):
                if pred_vi[j] == pred_vi[i]:
                    mat_pred[i][j] = 0

    vi_matrices.append(mat_pred)

    # return vi_matrices
    return vi_matrices


class VertexPlotBase(PlotBase):
    def plot(self):
        """
        VertexPlotBase subclass version of plot to plot the vertex index matrices for both true
        and predicted.
        """

        # required parameters for vertex index plot base. Set in 'style' key in config
        required_params = {
            'figsize',
            'fontsize',
            'label_fontsize',
            'dpi',
            'show_entries',
            'show_percentages',
            'text_color_threshold',
        }

        # filter only the necessary parameters from the config file to plot the vertex matrix
        filtered_params = {
            key: value for key, value in self.config.style.items() if key in required_params
        }

        # extracting sample details and storing as a dictionary
        sample = ConfigDict(self.config.samples)

        # extracting data and processing it
        with h5py.File(sample.path, "r") as hdf_file:
            jet_num = self.config.jet_num

            # extract jet information
            ds_jet = hdf_file['jets']
            truth_isDisp = ds_jet['isDisplaced'][jet_num]
            keys_list = list(ds_jet.dtype.fields.keys())
            prob_isDisp = ds_jet[keys_list[-2]][jet_num]

            # extract track information
            ds_tfj = hdf_file[sample.df_name]

            # boolean array of valid tracks
            valid = ds_tfj['valid'][jet_num]

            # extract vertex index data
            true_vi_data = ds_tfj['truthVertexIndex'][jet_num][valid]
            pred_vi_data = ds_tfj['VertexIndex'][jet_num][valid]

            n = len(true_vi_data)

            # extract valid origin labels
            true_origin_data = ds_tfj['truthOriginLabel'][jet_num][valid]
            pred_pileup_data = ds_tfj['pileup'][jet_num][valid]
            pred_fake_data = ds_tfj['fake'][jet_num][valid]
            pred_prompt_data = ds_tfj['prompt'][jet_num][valid]
            pred_disp_data = ds_tfj['displaced'][jet_num][valid]

            # sort
            sorted_indices = np.argsort(true_vi_data)

            # sort both true and predicted arrays based on sorted truth vertex index array
            true_vi = true_vi_data[sorted_indices]
            pred_vi = pred_vi_data[sorted_indices]

            # sort track origin data
            true_origin = true_origin_data[sorted_indices]
            pred_pileup = pred_pileup_data[sorted_indices]
            pred_fake = pred_fake_data[sorted_indices]
            pred_prompt = pred_prompt_data[sorted_indices]
            pred_disp = pred_disp_data[sorted_indices]

            # create vertex index matrices
            mat_true, mat_pred = make_VImats(
                true_vi, 
                pred_vi, 
                pred_pileup, 
                pred_fake, 
                pred_prompt,
                pred_disp
            )

        # tick positions and labels (x and y share same labels)
        xyticks = np.arange(0,n,5)
        xyticks_labels = []
        for i in range(n):
            if i%5 == 0:
                xyticks_labels.append(str(i))
            else:
                xyticks_labels.append(" ")

        # construct the plot
        fig = plt.figure(figsize=filtered_params['figsize'], dpi=filtered_params['dpi'],)
        gs = gridspec.GridSpec(1,2)

        # creating the subplots
        ax_true = fig.add_subplot(gs[0,0])
        ax_pred = fig.add_subplot(gs[0,1])

        # plotting the truth and predicted vertex index matrices
        ax_true.imshow(mat_true, cmap='gray')
        ax_pred.imshow(mat_pred, cmap='gray')

        if n < 60:
            size = 1200/n
        else:
            # m
            size = 600/n
        colors = ['gray', 'darkred', 'forestgreen', 'deepskyblue']
        labels = []
        # plot the truth origin labels
        for i in range(n):
            if true_origin[i] == 0:
                if 'pileup' not in labels:
                    labels.append('pileup')
                    ax_true.scatter(i, i, color=colors[0], marker='o', s=size, label='pileup')
                else:
                    ax_true.scatter(i, i, color=colors[0], marker='o', s=size)
            elif true_origin[i] == 1:
                if 'fake' not in labels:
                    labels.append('fake')
                    ax_true.scatter(i, i, color=colors[1], marker='o', s=size, label='fake')
                else:
                    ax_true.scatter(i, i, color=colors[1], marker='o', s=size)
            elif true_origin[i] == 2:
                if 'prompt' not in labels:
                    labels.append('prompt')
                    ax_true.scatter(i, i, color=colors[2], marker='o', s=size, label='prompt')
                else:    
                    ax_true.scatter(i, i, color=colors[2], marker='o', s=size)
            elif true_origin[i] == 3:
                if 'displaced' not in labels:
                    labels.append('displaced')
                    ax_true.scatter(i, i, color=colors[3], marker='o', s=size, label='displaced')
                else:
                    ax_true.scatter(i, i, color=colors[3], marker='o', s=size)
            else:
                print("ERROR: write an error message later")

        # add origin information to truth vertex index matrix
        for i, (pu, fk, pr, dp) in enumerate(zip(pred_pileup, pred_fake, pred_prompt, pred_disp)):
            origin = max(pu, fk, pr, dp)
            if origin == pu:
                ax_pred.scatter(i, i, color=colors[0], marker='o', s=size)
            elif origin == fk:
                ax_pred.scatter(i, i, color=colors[1], marker='o', s=size)
            elif origin == pr:
                ax_pred.scatter(i, i, color=colors[2], marker='o', s=size)
            elif origin == dp:
                ax_pred.scatter(i, i, color=colors[3], marker='o', s=size)
            else:
                print("ERROR: write an error message later")

        # tick positions and labels (x and y share same labels)
        if n <= 51:
	        xyticks = np.arange(0,n+1,5)
	        xyticks_labels = []
	        for i in range(n):
	            if i%5 == 0:
	                xyticks_labels.append(str(i))
	            else:
	                xyticks_labels.append(" ")
        else:
            xyticks = np.arange(0,n+1,10)
            xyticks_labels = []
            for i in range(n):
                if i%5 == 0:
                    xyticks_labels.append(str(i))
                else:
                    xyticks_labels.append(" ")

        # true vertex index matrix plot settings
        fsize = filtered_params['fontsize']
        ax_true.set_title("Truth Labels", fontsize=fsize)
        ax_true.set_xticks(xyticks, fontsize=fsize)
        ax_true.set_yticks(xyticks, fontsize=fsize)
        ax_true.set_xlabel(r"$n_{track}$", fontsize=fsize)
        ax_true.set_ylabel(r"$n_{track}$", fontsize=fsize)
        ax_true.legend(loc="upper right", fontsize=fsize-3)

        # predicted vertex index matrix plot settings
        ax_pred.set_title("GN2ej Prediction", fontsize=fsize)
        ax_pred.set_xticks(xyticks, fontsize=fsize)
        ax_pred.set_yticks(xyticks, fontsize=fsize)
        ax_pred.set_xlabel(r"$n_{track}$", fontsize=fsize)
        ax_pred.set_ylabel(r"$n_{track}$", fontsize=fsize)

        # add text to figure
        text = fig.text(0.93, 0.65, "Sample Jet {0}"
            "\n"
            "\n"
            "Truth {1} Jet"
            "\n"
            "\n"
            r"$P(Disp.)=${2:.3f}".format(jet_num, "Disp." if truth_isDisp == 1 else "Prompt", prob_isDisp),
            va='top', ha='center', fontsize=fsize-3, backgroundcolor='gray', alpha=1,

        )
        text.set_bbox(dict(facecolor='gray', alpha=0.25, linewidth=0))

        plt.tight_layout(rect=[0,0,0.86,1])

        plt.savefig(f"jet{jet_num}-VImatrices.png", dpi=filtered_params['dpi'], bbox_inches='tight')


