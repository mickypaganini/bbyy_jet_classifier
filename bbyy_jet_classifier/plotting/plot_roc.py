import cPickle
import logging
import os
import matplotlib.pyplot as plt
import plot_atlas
from viz import add_curve, calculate_roc, ROC_plotter
from ..utils import ensure_directory


def signal_eff_bkg_rejection(ML_strategy, yhat_test, test_data, yhat_old, sample_name):
    """
    Definition:
    -----------
        Check performance of ML_strategy by plotting its ROC curve and comparing it with the points generated by the old strategies

    Args:
    -----
        ML_strategy = one of the machine learning strategy in strategies/ whose prerformance we want to visualize
        yhat_test = array of dim (# testing examples), with predictions from the ML_strategy
        test_data = dictionary, containing "y", "w" for the test set, where:
            y = array of dim (# testing examples) with target values
            w = array of dim (# testing examples) with event weights
        yhat_old = dictionary containing predictions from the old strategies "mHmatch" and "pThigh", where:
            mHmatch_test = array of dim (# testing examples), containing the binary decision based the "closest mH" strategy
            pThigh_test  = array of dim (# testing examples), containing the binary decision based the "highest pT" strategy
    """
    # -- Calculate efficiencies from the older strategies
    # -- TO BE IMPROVED
    eff_mH_signal = float(sum((yhat_old["mHmatch"] * test_data["w"])[test_data["y"] == 1])) / float(sum(test_data["w"][test_data["y"] == 1])) if (sum(test_data["y"] == 1) > 0) else 0
    eff_mH_bkg = float(sum((yhat_old["mHmatch"] * test_data["w"])[test_data["y"] == 0])) / float(sum(test_data["w"][test_data["y"] == 0]))
    eff_pT_signal = float(sum((yhat_old["pThigh"] * test_data["w"])[test_data["y"] == 1])) / float(sum(test_data["w"][test_data["y"] == 1])) if (sum(test_data["y"] == 1) > 0) else 0
    eff_pT_bkg = float(sum((yhat_old["pThigh"] * test_data["w"])[test_data["y"] == 0])) / float(sum(test_data["w"][test_data["y"] == 0]))
    eff_pTjb_signal = float(sum((yhat_old["pTjb"] * test_data["w"])[test_data["y"] == 1])) / float(sum(test_data["w"][test_data["y"] == 1])) if (sum(test_data["y"] == 1) > 0) else 0
    eff_pTjb_bkg = float(sum((yhat_old["pTjb"] * test_data["w"])[test_data["y"] == 0])) / float(sum(test_data["w"][test_data["y"] == 0]))

    # -- Write old strategies pickle to output directory
    ensure_directory(os.path.join(ML_strategy.output_directory, "pickles", sample_name))
    cPickle.dump({
        "eff_mH_signal": eff_mH_signal,
        "eff_mH_bkg": eff_mH_bkg,
        "eff_pT_signal": eff_pT_signal,
        "eff_pT_bkg": eff_pT_bkg,
        "eff_pTjb_signal": eff_pTjb_signal,
        "eff_pTjb_bkg": eff_pTjb_bkg,
    }, open(os.path.join(ML_strategy.output_directory, "pickles", sample_name, "ROC_old_strategies_dict.pkl"), "wb"))

    # -- Initialise figure and axes
    logging.getLogger("plot_roc").info("Plotting signal efficiency and background rejection")
    plot_atlas.set_style()
    discrim_dict = add_curve(ML_strategy.name, "black", calculate_roc(test_data["y"], yhat_test, weights=test_data["w"]))
    figure = ROC_plotter(discrim_dict, min_eff=0.0, max_eff=1.0, min_rej=1, max_rej=10**4, logscale=True)

    # -- Add ROC curves and efficiency points for old strategies
    plt.plot(eff_mH_signal, 1.0 / eff_mH_bkg, marker="o", color="r", label=r"Closest m$_{H}$", linewidth=0)  # add point for "mHmatch" strategy
    plt.plot(eff_pT_signal, 1.0 / eff_pT_bkg, marker="o", color="b", label=r"Highest p$_{T}$", linewidth=0)  # add point for "pThigh" strategy
    plt.plot(eff_pTjb_signal, 1.0 / eff_pTjb_bkg, marker="o", color="magenta", label=r"Highest p$_{T, jb}$", linewidth=0)  # add point for "pTjb" strategy
    plt.legend()
    plot_atlas.use_atlas_labels(plt.axes())
    ensure_directory(os.path.join(ML_strategy.output_directory, "testing", sample_name))
    figure.savefig(os.path.join(ML_strategy.output_directory, "testing", sample_name, "ROC_{}_{}.pdf".format(ML_strategy.name, sample_name)))
    plt.close(figure)

    # -- Save ROC curve as pickle for later comparison
    cPickle.dump(discrim_dict[ML_strategy.name], open(os.path.join(ML_strategy.output_directory, "pickles", sample_name, "ROC_{}.pkl".format(ML_strategy.name)), "wb"), cPickle.HIGHEST_PROTOCOL)


def roc_comparison(ML_strategy, sample_name):
    """
    Definition:
    ------------
        Quick script to load and compare ROC curves produced from different classifiers
    """
    TMVABDT = cPickle.load(open(os.path.join(ML_strategy.output_directory, "pickles", sample_name, "ROC_root_tmva.pkl"), "rb"))
    sklBDT = cPickle.load(open(os.path.join(ML_strategy.output_directory, "pickles", sample_name, "ROC_skl_BDT.pkl"), "rb"))
    dots = cPickle.load(open(os.path.join(ML_strategy.output_directory, "pickles", sample_name, "ROC_old_strategies_dict.pkl"), "rb"))

    sklBDT["color"] = "green"
    curves = {"sklBDT": sklBDT, "RootTMVA": TMVABDT}

    # -- Initialise figure and axes
    logging.getLogger("plot_roc").info("Comparing {} strategies for {}".format(len(curves)+len(dots)/2, sample_name))
    figure = ROC_plotter(curves, title=r"Performance of Second b-Jet Selection Strategies", min_eff=0, max_eff=1.0, max_rej=10**4, logscale=True)

    plt.plot(dots["eff_mH_signal"], 1.0 / dots["eff_mH_bkg"], marker="o", color="r", label=r"Closest m$_{H}$", linewidth=0)  # add point for "mHmatch" strategy
    plt.plot(dots["eff_pT_signal"], 1.0 / dots["eff_pT_bkg"], marker="o", color="b", label=r"Highest p$_{T}$", linewidth=0)  # add point for "pThigh" strategy
    plt.plot(dots["eff_pTjb_signal"], 1.0 / dots["eff_pTjb_bkg"], marker="o", color="magenta", label=r"Highest p$_{T, jb}$", linewidth=0)  # add point for "pTjb" strategy

    plt.legend()
    plot_atlas.use_atlas_labels(plt.axes())
    ensure_directory(os.path.join(ML_strategy.output_directory, "testing", sample_name))
    figure.savefig(os.path.join(ML_strategy.output_directory, "testing", sample_name, "ROC_comparison_{}.pdf".format(sample_name)))
    plt.close(figure)
