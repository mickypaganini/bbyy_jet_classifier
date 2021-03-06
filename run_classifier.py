#! /usr/bin/env python
import argparse
import cPickle
import logging
import os
from bbyy_jet_classifier import strategies, process_data, utils
from bbyy_jet_classifier.plotting import plot_inputs, plot_outputs, plot_roc
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Run ML algorithms over ROOT TTree input")

    parser.add_argument("--input", required=True, type=str, nargs="+", 
        help="List of input file names")
    parser.add_argument("--tree", type=str, default="events_1tag",
        help="Name of the tree in the ntuples. Default: events_1tag")
    parser.add_argument("--output", type=str, default="output",
        help="Output directory. Default: output")
    parser.add_argument("--exclude", type=str, nargs="+", default=[], metavar="VARIABLE_NAME", 
        help="List of variables that are present in the tree but should not be used by the classifier")
    parser.add_argument("--strategy", type=str, nargs="+", default=["RootTMVA", "sklBDT"], 
        help="Type of BDT to use. Options are: RootTMVA, sklBDT. Default: both")
    parser.add_argument("--grid_search", action='store_true', 
        help="Pass this flag to run a grid search to determine BDT parameters")
    parser.add_argument("--ftrain", type=float, default=0.6, 
        help="Fraction of events to use for training. Default: 0.6. Set to 0 for testing only.")
    parser.add_argument("--training_sample", type=str, 
        help="Directory with pre-trained BDT to be used for testing")
    parser.add_argument("--max_events", type=int, default=-1, 
        help="Maximum number of events to use (for debugging). Default: all")

    return parser.parse_args()


def check_args(parsed_args):
    """
    Check the logic of the input arguments
    """
    if (parsed_args.ftrain < 0) or (parsed_args.ftrain > 1):
        raise ValueError("ftrain can only be a float between 0.0 and 1.0")

    if (parsed_args.ftrain == 0) and (parsed_args.training_sample is None):
        raise ValueError("When testing on 100% of the input file you need to specify which classifier to load. \
            Pass the folder containing the classifier to --training_sample.")

    if (parsed_args.ftrain > 0) and (parsed_args.training_sample is not None):
        raise ValueError("Training location is only a valid argument when ftrain == 0, because if you are using \
            {}% of your input data for training, you should not be testing on a separate pre-trained \
            classifier.".format(100 * parsed_args.ftrain))

    if not "output" in parsed_args.output :
        parsed_args.output = os.path.join("output", parsed_args.output)

    return parsed_args

# --------------------------------------------------------------

if __name__ == "__main__":

    # -- Configure logging
    utils.configure_logging()
    logger = logging.getLogger("run_classifier")

    # -- Parse arguments
    args = parse_args()
    args = check_args(args)

    # -- Construct list of input samples and dictionaries of sample -> data
    input_samples, training_data, testing_data, yhat_old_test_data, test_events_data = [], {}, {}, {}, {}

    # -- Ensure that output directory exists
    logger.info("Preparing output directory: {}".format(args.output))
    utils.ensure_directory(os.path.join(args.output))

    # -- Check that input file exists
    logger.info("Preparing to run over {} input samples".format(len(args.input)))
    for input_filename in args.input:
        logger.info("Now considering input: {}".format(input_filename))
        if not os.path.isfile(input_filename):
            raise OSError("{} does not exist!".format(args.input))

        # -- Set up folder paths
        input_samples.append(os.path.splitext(os.path.split(input_filename)[-1])[0])

        # -- Load in root files and return literally everything about the data
        classification_variables, variable2type,\
            training_data[input_samples[-1]], testing_data[input_samples[-1]],\
            yhat_old_test_data[input_samples[-1]], test_events_data[input_samples[-1]]\
            = process_data.load(input_filename, args.tree, args.exclude, args.ftrain, args.max_events)

        #-- Plot input distributions
        plot_inputs.input_distributions(classification_variables, training_data[input_samples[-1]], testing_data[input_samples[-1]],
                                        output_directory=os.path.join(args.output, "classification_variables", input_samples[-1]))

    # -- Combine all training data into a new sample
    if args.ftrain > 0:
        train_data_combined = process_data.combine_datasets(training_data.values())
        logger.info("Combining {} into one training sample with {} jet pairs".format(len(input_samples), train_data_combined["y"].shape[0]))
    combined_input_sample = input_samples[0] if len(input_samples) == 1 else "merged_inputs"
    old_strategy_names = yhat_old_test_data[input_samples[0]].keys()

    # -- Sequentially evaluate all the desired strategies on the same train/test sample
    for strategy_name in args.strategy:

        # -- Construct dictionary of available strategies
        if strategy_name not in strategies.__dict__.keys():
            raise AttributeError("{} is not a valid strategy".format(strategy_name))
        ML_strategy = getattr(strategies, strategy_name)(args.output)

        # -- Training!
        if args.ftrain > 0:
            logger.info("Preparing to train with {}% of events and then test with the remainder".format(int(100 * args.ftrain)))

            # -- Train classifier
            ML_strategy.train(train_data_combined, classification_variables, variable2type, sample_name=combined_input_sample, grid_search=args.grid_search)

            # -- Plot the classifier output as tested on each of the training sets
            # -- (only useful if you care to check the performance on the training set)
            for (sample_name, train_data) in training_data.items():
                logger.info("Sanity check: testing global classifier output on training set {}".format(sample_name))
                yhat_train, yhat_class_train = ML_strategy.test(train_data, classification_variables, training_sample=combined_input_sample)
                plot_outputs.classifier_output(ML_strategy, yhat_train, train_data, process="training", sample_name=sample_name)

        else:
            logger.info("Preparing to use 100% of sample as testing input")

        # -- Testing!
        if args.ftrain < 1:
            logger.info("Preparing to test with {}% of events".format(int(100 * (1-args.ftrain))))
            training_sample = args.training_sample if args.training_sample is not None else combined_input_sample

            for (sample_name, test_data) in testing_data.items():
                logger.info("Running classifier from {} on testing set for {}".format(training_sample, sample_name))

                # -- Test classifier
                yhat_test, yhat_class_test = ML_strategy.test(test_data, classification_variables, training_sample=training_sample)

                # -- Plot output testing distributions from classifier
                plot_outputs.classifier_output(ML_strategy, yhat_test, test_data, process="testing", sample_name=sample_name)
                plot_outputs.confusion(ML_strategy, yhat_class_test, test_data, ML_strategy.name, sample_name=sample_name)

                # -- Plot yhat output for the old strategies
                for old_strategy_name in old_strategy_names:
                    plot_outputs.old_strategy(ML_strategy, yhat_old_test_data[sample_name][old_strategy_name] == 0, test_data, old_strategy_name, sample_name=sample_name)
                    plot_outputs.confusion(ML_strategy, yhat_old_test_data[sample_name][old_strategy_name] == 0, test_data, old_strategy_name, sample_name=sample_name)

                # -- Performance evaluation:
                # -- 1) Jet-pair level

                # -- Visualize performance by displaying the ROC curve from the selected ML strategy and comparing it with the old strategies
                logger.info("Plotting ROC curve from {} for {} sample".format(ML_strategy.name, sample_name))
                yhat_old_test_data_dict = dict( [(old_strategy_name, yhat_old_test_data[sample_name][old_strategy_name] == 0) for old_strategy_name in old_strategy_names] )
                plot_roc.signal_eff_bkg_rejection(ML_strategy, yhat_test, test_data, yhat_old_test_data_dict, sample_name)

                # -- 2) Event level
                #       y will be used to match event-level shape from flattened arrays
                #       m_jb will be used to check if the selected jet pair falls into the m_jb mass window
                #       pT_j will be used to try cutting on the jet pTs

                # -- put arrays back into event format by matching shape of y_event
                yhat_test_ev = {"classifier": process_data.match_shape(yhat_test, test_events_data[sample_name]["y"])}
                for old_strategy_name in old_strategy_names:
                    yhat_test_ev[old_strategy_name] = process_data.match_shape(yhat_old_test_data_dict[old_strategy_name], test_events_data[sample_name]["y"])

                # -- print performance
                logger.info("Writing out event-level performance information...")
                utils.ensure_directory(os.path.join(ML_strategy.output_directory, "pickles", sample_name))
                cPickle.dump({"yhat_test_ev": yhat_test_ev["classifier"],
                              "yhat_mHmatch_test_ev": yhat_test_ev["mHmatch"],
                              "yhat_pThigh_test_ev": yhat_test_ev["pThigh"],
                              "yhat_pTjb_test_ev": yhat_test_ev["pTjb"],
                              "y_event": test_events_data[sample_name]["y"],
                              "mjb_event": test_events_data[sample_name]["m_jb"],
                              "pTj_event": test_events_data[sample_name]["pT_j"],
                              "w_test": test_events_data[sample_name]["w"]},
                             open(os.path.join(ML_strategy.output_directory, "pickles", sample_name, "{}_event_performance_dump.pkl".format(ML_strategy.name)), "wb"), cPickle.HIGHEST_PROTOCOL)

        else:
            logger.info("100% of the sample was used for training -- no independent testing can be performed.")

    # -- if there is more than one strategy and we aren't only training, plot the ROC comparison
    if len(args.strategy) > 1 and (args.ftrain < 1):
        logger.info("Plotting ROC comparisons for {} samples".format(len(input_samples)))
        for input_sample in input_samples:
            plot_roc.roc_comparison(ML_strategy, input_sample)
