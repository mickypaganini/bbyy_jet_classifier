#! /usr/bin/env python
import argparse
import cPickle as pickle
import logging
import os
from bbyy_jet_classifier import strategies, process_data, utils
from bbyy_jet_classifier.plotting import plot_inputs, plot_outputs, plot_roc
import numpy as np

FTRAIN = 0.7
# TO DO:
# - remove excluded variables
# - remove events using max_events
# - load train and test separately


def parse_args():
    parser = argparse.ArgumentParser(description="Run ML classifiers for 2nd jet selection")
    # -- can either train or test

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--input", required=True, type=str, nargs="+", 
                               help="List of original ROOT input file paths")
    parent_parser.add_argument("--tree",
        type=str, default="events_1tag",
        help="Name of the tree in the ntuples. Default: events_1tag")
    parent_parser.add_argument("--strategy",
        type=str, nargs="+", default=["RootTMVA", "sklBDT"], 
        help="Type of BDT to use. Options are: RootTMVA, sklBDT. Default: both")
    parent_parser.add_argument("--exclude",
        type=str, nargs="+", default=[], metavar="VARIABLE_NAME", 
        help="List of variables that are present in the tree but should not be used by the classifier")
    parent_parser.add_argument("--max_events",
        type=int, default=-1, 
        help="Maximum number of events to use (for debugging). Default: all")
    parent_parser.add_argument("--output",
        type=str, default="output",
        help="Output directory. Default: output") # needed?
    parent_parser.add_argument("--training_id", type=str,
        required=True,
        help="String to identify training in the future or to select which pretrained model to test")

    subparsers = parser.add_subparsers(title="actions", dest='action')
    parser_train = subparsers.add_parser("train", parents=[parent_parser],
        help="train BDT for jet-pair classification")
    # parser_train.add_argument("--ftrain",
    #     type=float, default=0.7, 
    #     help="Fraction of events to use for training. Default: 0.6.")
    parser_train.add_argument("--grid_search",
        action='store_true', 
        help="Pass this flag to run a grid search to determine BDT parameters")

    parser_test = subparsers.add_parser("test", parents=[parent_parser],
        help="test BDT for jet-pair classification")

    args = parser.parse_args()
   
    return args


def check_args(parsed_args):
    """
    Check the logic of the input arguments
    """
    # if parsed_args.action == 'train':
    #     if (parsed_args.ftrain < 0) or (parsed_args.ftrain > 1):
    #         raise ValueError("ftrain can only be a float between 0.0 and 1.0")

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

    # -- Ensure that output directory exists
    logger.info("Preparing output directory: {}".format(args.output))
    utils.ensure_directory(os.path.join(args.output))

    # -- Construct list of input samples and dictionaries of sample -> data
    input_samples = []
    training_data = {}
    testing_data = {}
    yhat_old_test_data = {}
    test_events_data = {}

    # -- Check that input file exists
    logger.info("Preparing to run over {} input samples".format(len(args.input)))
    for input_filename in args.input:
        logger.info("Now considering input: {}".format(input_filename))
        if not os.path.isfile(input_filename):
            raise OSError("{} does not exist!".format(args.input))

        # -- Set up folder paths
        input_samples.append(os.path.splitext(os.path.split(input_filename)[-1])[0])

        # -- Load in root files and return literally everything about the data
        pklpath = os.path.join(os.path.split(input_filename)[0], 
            '{fname}---{tree}---{frac}---train.pkl'.format(
                fname=os.path.split(input_filename)[1],
                tree=args.tree, 
                frac=str(FTRAIN)
            )
        )

        try:
            if args.action == 'train':
                logger.info('Trying to load data from ' + pklpath)
                train_pkl = pickle.load(open(pklpath, 'rb'))
                classification_variables = train_pkl['classification_variables']
                variable2type = train_pkl['variable2type']
                training_data[input_samples[-1]] = train_pkl['train_data']
                logger.info('Data found and loaded from ' + pklpath)
                if args.exclude:
                    training_data[input_samples[-1]]['X'] = training_data[input_samples[-1]]['X'][:, np.array(
                        [i for i, entry in enumerate(classification_variables) if entry not in args.exclude]
                    )]

                # logger.info('TRAIN_SHAPE = {}'.format(training_data[input_samples[-1]].shape))


            elif args.action == 'test':
                pklpath = pklpath.replace('---train.pkl', '---test.pkl')
                logger.info('Trying to load data from ' + pklpath)
                test_pkl = pickle.load(open(pklpath, 'rb'))   
                classification_variables = test_pkl['classification_variables']
                variable2type = test_pkl['variable2type']
                testing_data[input_samples[-1]] = test_pkl['test_data']
                yhat_old_test_data[input_samples[-1]] = test_pkl['yhat_old_test_data']
                test_events_data[input_samples[-1]] = test_pkl['test_events_data']
                logger.info('Data found and loaded from ' + pklpath)
                if args.exclude:
                    testing_data[input_samples[-1]]['X'] = testing_data[input_samples[-1]]['X'][:, np.array(
                        [i for i, entry in enumerate(classification_variables) if entry not in args.exclude]
                    )]
        except Exception, e:
            logger.info('No valid data found in {}. Reprocessing... '.format(pklpath))
            classification_variables, variable2type,\
                training_data[input_samples[-1]], testing_data[input_samples[-1]],\
                yhat_old_test_data[input_samples[-1]], test_events_data[input_samples[-1]]\
                = process_data.load(input_filename, args.tree, FTRAIN, pklpath)
            #-- Plot input distributions
            plot_inputs.input_distributions(
                classification_variables, training_data[input_samples[-1]], testing_data[input_samples[-1]],
                output_directory=os.path.join(args.output, "classification_variables", input_samples[-1])
            )
            testing_data[input_samples[-1]]['X'] = testing_data[input_samples[-1]]['X'][:, np.array(
                [i for i, entry in enumerate(classification_variables) if entry not in args.exclude]
            )]

            training_data[input_samples[-1]]['X'] = training_data[input_samples[-1]]['X'][:, np.array(
                [i for i, entry in enumerate(classification_variables) if entry not in args.exclude]
            )]



        if args.exclude:
            classification_variables = [entry for entry in classification_variables if entry not in args.exclude]
            for k in args.exclude:
                del variable2type[k]
    
    if args.action == 'train':
        # -- Combine all training data into a new sample

        for d in training_data.values():
            logger.info('LOOK -> {}'.format(d['X'].shape))
        train_data_combined = process_data.combine_datasets(training_data.values()) # added shuffling

        if args.max_events != -1:
            train_data_combined = {k : v[:args.max_events] for k,v in train_data_combined.iteritems()}
            # could be moved into function above

        logger.info("Combining {} into one training sample with {} jet pairs".format(len(input_samples), train_data_combined["y"].shape[0]))
        combined_input_sample = args.training_id #input_samples[0] if len(input_samples) == 1 else "merged_inputs"


        for strategy_name in args.strategy:
            # -- Construct dictionary of available strategies
            if strategy_name not in strategies.__dict__.keys():
                raise AttributeError("{} is not a valid strategy".format(strategy_name))
            ML_strategy = getattr(strategies, strategy_name)(args.output)

            # -- Train classifier
            ML_strategy.train(
                train_data_combined, classification_variables, variable2type, sample_name=combined_input_sample, grid_search=args.grid_search
            )
            # -- Plot the classifier output as tested on each of the training sets
            # -- (only useful if you care to check the performance on the training set)
            # -- NOTE: checking on entire training set even if we passed max_events (potentially to be changed)
            for (sample_name, train_data) in training_data.items():
                logger.info("Sanity check: testing global classifier output on training set {}".format(sample_name))
                logger.info('SHAPE = {}'.format(train_data['X'].shape))
                yhat_train, yhat_class_train = ML_strategy.test(
                    train_data, classification_variables, training_sample=combined_input_sample
                )
                plot_outputs.classifier_output(
                    ML_strategy, yhat_train, train_data, process="training", sample_name=sample_name
                )

    elif args.action == 'test':
        old_strategy_names = yhat_old_test_data[input_samples[0]].keys()
        for strategy_name in args.strategy:
            # -- Construct dictionary of available strategies
            if strategy_name not in strategies.__dict__.keys():
                raise AttributeError("{} is not a valid strategy".format(strategy_name))
            ML_strategy = getattr(strategies, strategy_name)(args.output)

            training_sample = args.training_id #args.training_sample if args.training_sample is not None else combined_input_sample

            for (sample_name, test_data) in testing_data.items():
                logger.info("Running classifier from {} on testing set for {}".format(training_sample, sample_name))

                # -- Test classifier
                yhat_test, yhat_class_test = ML_strategy.test(
                    test_data, classification_variables, training_sample=training_sample
                )

                # -- Plot output testing distributions from classifier
                plot_outputs.classifier_output(
                    ML_strategy, yhat_test, test_data, process="testing", sample_name=sample_name
                )
                plot_outputs.confusion(
                    ML_strategy, yhat_class_test, test_data, ML_strategy.name, sample_name=sample_name
                )

                # -- Plot yhat output for the old strategies
                for old_strategy_name in old_strategy_names:
                    plot_outputs.old_strategy(
                        ML_strategy, yhat_old_test_data[sample_name][old_strategy_name] == 0, test_data, old_strategy_name, sample_name=sample_name
                    )
                    plot_outputs.confusion(
                        ML_strategy, yhat_old_test_data[sample_name][old_strategy_name] == 0, test_data, old_strategy_name, sample_name=sample_name
                    )

                # -- Performance evaluation:
                # -- 1) Jet-pair level

                # -- Visualize performance by displaying the ROC curve from the selected ML strategy and comparing it with the old strategies
                logger.info("Plotting ROC curve from {} for {} sample".format(ML_strategy.name, sample_name))
                yhat_old_test_data_dict = dict([
                    (old_strategy_name, yhat_old_test_data[sample_name][old_strategy_name] == 0) 
                    for old_strategy_name in old_strategy_names
                ])
                plot_roc.signal_eff_bkg_rejection(ML_strategy, yhat_test, test_data, yhat_old_test_data_dict, sample_name)

                # -- 2) Event level
                #       y will be used to match event-level shape from flattened arrays
                #       m_jb will be used to check if the selected jet pair falls into the m_jb mass window
                #       pT_j will be used to try cutting on the jet pTs

                # -- put arrays back into event format by matching shape of y_event
                yhat_test_ev = {
                    "classifier": process_data.match_shape(yhat_test, test_events_data[sample_name]["y"])
                }
                for old_strategy_name in old_strategy_names:
                    yhat_test_ev[old_strategy_name] = process_data.match_shape(
                        yhat_old_test_data_dict[old_strategy_name], test_events_data[sample_name]["y"]
                )

                # -- print performance
                logger.info("Writing out event-level performance information...")
                utils.ensure_directory(os.path.join(ML_strategy.output_directory, "pickles", sample_name))
                pickle.dump({"yhat_test_ev": yhat_test_ev["classifier"],
                              "yhat_mHmatch_test_ev": yhat_test_ev["mHmatch"],
                              "yhat_pThigh_test_ev": yhat_test_ev["pThigh"],
                              "yhat_pTjb_test_ev": yhat_test_ev["pTjb"],
                              "y_event": test_events_data[sample_name]["y"],
                              "mjb_event": test_events_data[sample_name]["m_jb"],
                              "pTj_event": test_events_data[sample_name]["pT_j"],
                              "w_test": test_events_data[sample_name]["w"]},
                            open(os.path.join(
                                ML_strategy.output_directory, "pickles", sample_name, "{}_event_performance_dump.pkl".format(ML_strategy.name)
                            ),
                            "wb"), pickle.HIGHEST_PROTOCOL
                )

    else:
        raise ValueError('Only valid actions are "train" and "test"')


    # -- if there is more than one strategy and we aren't only training, plot the ROC comparison
    if len(args.strategy) > 1 and ('test' in args.action):
        logger.info("Plotting ROC comparisons for {} samples".format(len(input_samples)))
        for input_sample in input_samples:
            plot_roc.roc_comparison(ML_strategy, input_sample)
