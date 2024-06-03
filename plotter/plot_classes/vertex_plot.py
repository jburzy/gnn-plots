from puma.matshow import MatshowPlot
from plotter.config_dict import ConfigDict
import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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
            mat_pred[i][i] == 0

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
            'xlabel',
            'ylabel',
            'atlas_second_tag',
            'figsize',
            'y_scale',
            'fontsize',
            'label_fontsize',
            'dpi',
            'show_entries',
            'show_percentages',
            'text_color_threshold',
            'colormap',
            'cbar_label',
            'use_atlas_tag'
        }

        # filter only the necessary parameters from the config file to plot the vertex matrix
        filtered_params = {
            key: value for key, value in self.config.style.items() if key in required_params
        }

        # constructing vertex index matrices for both truth and model prediction
        sample = ConfigDict(self.config.samples)
        with h5py.File(sample.path, "r") as hdf_file:
            jet_num = self.config.jet_num

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

            # take valid tracks from truth origin label (i.e. tracks not coming from -1 entry) and sort
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

        # create plot base for truth vertex index matrix
        true_vi_plot = MatshowPlot(
            x_ticklabels=xyticks_labels,
            x_ticks_rotation=0,
            y_ticklabels=xyticks_labels,
            show_entries=filtered_params['show_entries'],
            show_percentage=filtered_params['show_percentages'],
            colormap=filtered_params['colormap'],
            text_color_threshold=filtered_params['text_color_threshold'],
            cbar_label=filtered_params['cbar_label'],
            xlabel=rf"{filtered_params['xlabel']}",
            ylabel=rf"{filtered_params['ylabel']}",
            fontsize=filtered_params['fontsize'],
            label_fontsize=filtered_params['label_fontsize'],
            figsize=filtered_params['figsize'],
            dpi=filtered_params['dpi'],
            use_atlas_tag=filtered_params['use_atlas_tag']
        )

        # draw true vertex index matrix
        true_vi_plot.draw(mat_true)

        colors = ['gray', 'darkred', 'forestgreen', 'deepskyblue']

        # add origin information to truth vertex index matrix
        for i in range(n):
            if true_origin[i] == 0:
                true_vi_plot.axis_top.scatter(i, i, color=colors[0], marker='o', s=30)
            elif true_origin[i] == 1:
                true_vi_plot.axis_top.scatter(i, i, color=colors[1], marker='o', s=30)
            elif true_origin[i] == 2:
                true_vi_plot.axis_top.scatter(i, i, color=colors[2], marker='o', s=30)
            elif true_origin[i] == 3:
                true_vi_plot.axis_top.scatter(i, i, color=colors[3], marker='o', s=30)
            else:
                print("ERROR: write an error message later")

        # set title
        true_vi_plot.set_title(f"Truth Vertex Index Matrix, Jet {jet_num}")
        
        # # Saving the plot
        true_vi_plot.savefig(f"jet{jet_num}-trueVImatrix")

        # create plot base for predicted vertex index matrix
        pred_vi_plot = MatshowPlot(
            x_ticklabels=xyticks_labels,
            x_ticks_rotation=0,
            y_ticklabels=xyticks_labels,
            show_entries=filtered_params['show_entries'],
            show_percentage=filtered_params['show_percentages'],
            colormap=filtered_params['colormap'],
            text_color_threshold=filtered_params['text_color_threshold'],
            cbar_label=filtered_params['cbar_label'],
            xlabel=rf"{filtered_params['xlabel']}",
            ylabel=rf"{filtered_params['ylabel']}",
            fontsize=filtered_params['fontsize'],
            label_fontsize=filtered_params['label_fontsize'],
            figsize=filtered_params['figsize'],
            dpi=filtered_params['dpi'],
            use_atlas_tag=filtered_params['use_atlas_tag']
        )

        # draw true vertex index matrix
        pred_vi_plot.draw(mat_pred)

        # add origin information to truth vertex index matrix
        for i, (pu, fk, pr, dp) in enumerate(zip(pred_pileup, pred_fake, pred_prompt, pred_disp)):
            origin = max(pu, fk, pr, dp)
            if origin == pu:
                pred_vi_plot.axis_top.scatter(i, i, color=colors[0], marker='o', s=30)
            elif origin == fk:
                pred_vi_plot.axis_top.scatter(i, i, color=colors[1], marker='o', s=30)
            elif origin == pr:
                pred_vi_plot.axis_top.scatter(i, i, color=colors[2], marker='o', s=30)
            elif origin == dp:
                pred_vi_plot.axis_top.scatter(i, i, color=colors[3], marker='o', s=30)
            else:
                print("ERROR: write an error message later")

        # set title
        pred_vi_plot.set_title(f"Predicted Vertex Index Matrix, Jet {jet_num}")
            
        # Saving the plot
        pred_vi_plot.savefig(f"jet{jet_num}-predVImatrix")
