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
            # still have to check if these tracks are predicted to pair with other tracks
            pair = False
            for j in range(n):
                if pred_vi[j] == pred_vi[i]:
                    mat_pred[i][j] = 0
                    pair = True
            if pair == True:
                # give it a different value from 0 to distinguish
                # NOTE: THIS IS HERE IF I WANT TO ADD ANOTHER ITEM IN THE LEGEND FOR SHOWING THIS CASE
                mat_pred[i][i] = 0.01
            else:
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
        VertexPlotBase subclass of PlotBase to plot the vertex index matrices for both true
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


        # EXTRACTING THE DATA AND PROCESS IT
        # ----------------------------------
        with h5py.File(sample.path, "r") as hdf_file:
            jet_num = self.config.jet_num

            # extract jet information
            ds_jet = hdf_file['jets']
            truth_isDisp = ds_jet['isDisplaced'][jet_num]
            keys_list = list(ds_jet.dtype.fields.keys())
            prob_isDisp = ds_jet[keys_list[-2]][jet_num]
            jet_pt = ds_jet['pt'][jet_num]/1000     # jet transverse momentum in GeV
            jet_eta = ds_jet['eta'][jet_num]

            # extract track information
            ds_tfj = hdf_file[sample.df_name]

            # boolean array of valid tracks
            valid = ds_tfj['valid'][jet_num]

            # extract vertex index data
            true_vi_data = ds_tfj['truthVertexIndex'][jet_num][valid]
            pred_vi_data = ds_tfj['VertexIndex'][jet_num][valid]

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

            # set the view: two options are global and closeup
            if self.config.zoom:
                # adjust matrices based on when no pileup tracks are in truth sample
                for i in range(len(true_vi)):
                    if true_vi[i] != -2:
                        true_vi = true_vi[i:]
                        pred_vi = pred_vi[i:]
                        true_origin = true_origin[i:]
                        pred_pileup = pred_pileup[i:]
                        pred_fake = pred_fake[i:]
                        pred_prompt = pred_prompt[i:]
                        pred_disp = pred_disp[i:]
                        break

            n = len(true_vi)

            # create vertex index matrices
            mat_true, mat_pred = make_VImats(
                true_vi, 
                pred_vi, 
                pred_pileup, 
                pred_fake, 
                pred_prompt,
                pred_disp
            )


        # CONSTRUCTING THE FIGURE AND PLOTTING THE MATRICES
        # -------------------------------------------------
        fig = plt.figure(figsize=filtered_params['figsize'], dpi=filtered_params['dpi'],)
        gs = gridspec.GridSpec(1,2)

        # creating the subplots
        ax_true = fig.add_subplot(gs[0,0])
        ax_pred = fig.add_subplot(gs[0,1])

        # plotting the truth and predicted vertex index matrices
        ax_true.imshow(mat_true, cmap='gray')
        ax_pred.imshow(mat_pred, cmap='gray')


        # ADJUSTING PLOT SETTINGS
        # -----------------------
        # adjust the scatterplot sizes depending on number of valid tracks
        if n <= 30:
            size = 1600/n
        elif (n > 30) and (n < 60):
            size = 1200/n
        elif (n>=60) and (n<120):
            size = 600/n
        else:
            size = 300/n

        colors = ['gray', 'darkred', 'forestgreen', 'deepskyblue']
        # initialize labels to keep track of what label has been applied to the legend
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

        # initialize labels to keep track of what label has been applied to the legend
        labels = []

        # add origin information to truth vertex index matrix
        for i, (pu, fk, pr, dp) in enumerate(zip(pred_pileup, pred_fake, pred_prompt, pred_disp)):
            origin = max(pu, fk, pr, dp)
            if origin == pu:
                if 'pileup' not in labels:
                    labels.append('pileup')
                    ax_pred.scatter(i, i, color=colors[0], marker='o', s=size, label='pileup')
                else:
                    ax_pred.scatter(i, i, color=colors[0], marker='o', s=size)
            elif origin == fk:
                if 'fake' not in labels:
                    labels.append('fake')
                    ax_pred.scatter(i, i, color=colors[1], marker='o', s=size, label='fake')
                else:
                    ax_pred.scatter(i, i, color=colors[1], marker='o', s=size)
            elif origin == pr:
                if 'prompt' not in labels:
                    labels.append('prompt')
                    ax_pred.scatter(i, i, color=colors[2], marker='o', s=size, label='prompt')
                else:
                    ax_pred.scatter(i, i, color=colors[2], marker='o', s=size)
            elif origin == dp:
                if 'displaced' not in labels:
                    labels.append('displaced')
                    ax_pred.scatter(i, i, color=colors[3], marker='o', s=size, label='displaced')
                else:
                    ax_pred.scatter(i, i, color=colors[3], marker='o', s=size)
            else:
                print("ERROR: write an error message later")

        # tick positions and labels (x and y share same labels)
        if n <= 51:
            xyticks = np.arange(0,n+1,5)
        elif (n>51) and (n<=101):
            xyticks = np.arange(0,n+1,10)
        else:
            xyticks = np.arange(0,n+1,20)

        # true vertex index matrix plot settings
        fsize = filtered_params['fontsize']
        ax_true.set_title("Truth Labels", fontsize=fsize)
        ax_true.set_xticks(xyticks, fontsize=fsize)
        ax_true.set_yticks(xyticks, fontsize=fsize)
        ax_true.set_xlabel(r"$n_{track}$", fontsize=fsize)
        ax_true.set_ylabel(r"$n_{track}$", fontsize=fsize)

        # predicted vertex index matrix plot settings
        ax_pred.set_title("GN2ej Prediction", fontsize=fsize)
        ax_pred.set_xticks(xyticks, fontsize=fsize)
        ax_pred.set_yticks(xyticks, fontsize=fsize)
        ax_pred.set_xlabel(r"$n_{track}$", fontsize=fsize)
        ax_pred.set_ylabel(r"$n_{track}$", fontsize=fsize)

        # add text to figure
        text = fig.text(0.94, 0.35, "Sample Jet {0}"
            "\n"
            "\n"
            "Truth {1} Jet"
            "\n"
            "\n"
            r"Jet $p_{{T}}={2:.1f}$ GeV"
            "\n"
            "\n"
            r"Jet $\eta={3:.3f}$"
            "\n"
            "\n"
            r"$P_{{EJ}}=${4:.3f}".format(jet_num, "Signal" if truth_isDisp == 1 else "QCD", jet_pt, 
                jet_eta, prob_isDisp), ha='center', fontsize=fsize-3, backgroundcolor='gray', alpha=1,
        )
        # adjust border box of text
        text.set_bbox(dict(boxstyle='round', facecolor='gray', alpha=0.25, linewidth=0))

        # retrieve handles and labels for each plot
        handles, labels = ax_true.get_legend_handles_labels()
        handles_pred, labels_pred = ax_pred.get_legend_handles_labels()
        # append predicted matrix plot labels into labels if they are not already present
        for i, label in enumerate(labels_pred):
            if label not in labels:
                handles.append(handles_pred[i])
                labels.append(labels_pred[i])

        fig.legend(handles=handles, labels=labels, fontsize=fsize-3, loc='lower center', 
            bbox_to_anchor=(0.45,-0.03), fancybox=True, shadow=True, ncol=len(handles))

        plt.tight_layout(rect=[0,0,0.86,1])

        if self.config.zoom:
            plt.savefig(f"jet{jet_num}-VImatrices_zoom.png", dpi=filtered_params['dpi'], bbox_inches='tight')
        else:
            plt.savefig(f"jet{jet_num}-VImatrices.png", dpi=filtered_params['dpi'], bbox_inches='tight')

