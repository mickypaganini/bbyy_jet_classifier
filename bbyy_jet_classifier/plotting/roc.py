import logging
import os

import cPickle
import matplotlib.pyplot as plt
import numpy as np
import rootpy.plotting as rpp

from viz import add_curve, calculate_roc, ROC_plotter
import plot_atlas

def signal_eff_bkg_rejection(ML_strategy, mHmatch_test, pThigh_test, yhat_test, test_data):
    """
    Definition:
    -----------
        Check performance of ML_strategy by plotting its ROC curve and comparing it with the points generated by the old strategies

    Args:
    -----
        ML_strategy = one of the machine learning strategy in strategies/ whose prerformance we want to visualize
        mHmatch_test = array of dim (# testing examples), containing the binary decision based the "closest mH" strategy
        pThigh_test = array of dim (# testing examples), containing the binary decision based the "highest pT" strategy
        yhat_test = array of dim (# testing examples), with predictions from the ML_strategy
        test_data = dictionary, containing 'y', 'w' for the test set, where:
                y = array of dim (# testing examples) with target values
                w = array of dim (# testing examples) with event weights
    """
    rpp.set_style("ATLAS", mpl=True)
    logging.getLogger("Plotting").info("Plotting performance")

    # -- Calculate efficiencies from the older strategies
    eff_mH_signal = float(sum((mHmatch_test * test_data['w'])[test_data['y'] == 1])) / float(sum(test_data['w'][test_data['y']== 1]))
    eff_mH_bkg = float(sum((mHmatch_test * test_data['w'])[test_data['y'] == 0])) / float(sum(test_data['w'][test_data['y']== 0]))
    eff_pT_signal = float(sum((pThigh_test * test_data['w'])[test_data['y'] == 1])) / float(sum(test_data['w'][test_data['y']== 1]))
    eff_pT_bkg = float(sum((pThigh_test * test_data['w'])[test_data['y'] == 0])) / float(sum(test_data['w'][test_data['y']== 0]))

    ML_strategy.ensure_directory(os.path.join(ML_strategy.output_directory, "pickle"))
    cPickle.dump(
        {"eff_mH_signal": eff_mH_signal,
         "eff_mH_bkg": eff_mH_bkg,
         "eff_pT_signal": eff_pT_signal,
         "eff_pT_bkg": eff_pT_bkg
         }, open(os.path.join(ML_strategy.output_directory, "pickle", "old_strategies_dict.pkl"), "wb"))

    # -- Add ROC curves and efficiency points for old strategies
    discs = {}
    add_curve(ML_strategy.name, "black", calculate_roc(test_data['y'], yhat_test, weights=test_data['w']), discs)
    fg = ROC_plotter(discs, min_eff=0.1, max_eff=1.0, logscale=True)
    plt.plot(eff_mH_signal, 1.0 / eff_mH_bkg, marker="o", color="r", label=r"Closest m$_{H}$", linewidth=0)  # add point for "mHmatch" strategy
    plt.plot(eff_pT_signal, 1.0 / eff_pT_bkg, marker="o", color="b", label=r"Highest p$_{T}$", linewidth=0)  # add point for "pThigh" strategy
    plt.legend()
    axes = plt.axes()
    plot_atlas.atlaslabel(axes)
    fg.savefig(os.path.join(ML_strategy.output_directory, "ROC.pdf"))
    # -- Save out ROC curve as pickle for later comparison
    cPickle.dump(discs[ML_strategy.name], open(os.path.join(ML_strategy.output_directory, "pickle", "{}_ROC.pkl".format(ML_strategy.name)), "wb"), cPickle.HIGHEST_PROTOCOL)


def roc_comparison():
    '''
    Definition:
    ------------
        Quick script to load and compare ROC curves produced from different classifiers
    '''

    TMVABDT = cPickle.load(open(os.path.join("output", "RootTMVA", "pickle", "RootTMVA_ROC.pkl"), "rb"))
    sklBDT = cPickle.load(open(os.path.join("output", "sklBDT", "pickle", "sklBDT_ROC.pkl"), "rb"))
    dots = cPickle.load(open(os.path.join("output", "sklBDT", "pickle", "old_strategies_dict.pkl"), "rb"))

    sklBDT["color"] = "green"

    curves = {}
    curves["sklBDT"] = sklBDT
    curves["RootTMVA"] = TMVABDT

    logging.getLogger("RunClassifier").info("Plotting")
    fg = ROC_plotter(curves, title=r"Performance of Second b-Jet Selection Strategies", min_eff=0.1, max_eff=1.0, ymax=1000, logscale=True)
    plt.plot(dots["eff_mH_signal"], 1.0 / dots["eff_mH_bkg"], marker="o", color="r", label=r"Closest m$_{H}$", linewidth=0)  # add point for "mHmatch" strategy
    plt.plot(dots["eff_pT_signal"], 1.0 / dots["eff_pT_bkg"], marker="o", color="b", label=r"Highest p$_{T}$", linewidth=0)  # add point for "pThigh" strategy
    plt.legend()
    axes = plt.axes()
    plot_atlas.atlaslabel(axes, fontsize=10)
    fg.savefig(os.path.join("output", "ROCcomparison.pdf"))

