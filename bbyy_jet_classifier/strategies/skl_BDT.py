import cPickle
import logging
import os
from . import BaseStrategy
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.metrics import classification_report


class sklBDT(BaseStrategy):
    """
    Strategy using a BDT from scikit-learn
    """
    default_output_subdir = "sklBDT"  # os.path.join("output", "sklBDT")

    def train(self, train_data, classification_variables, variable_dict):
        """
        Definition:
        -----------
            Training method for sklBDT; it pickles the model into the "pickle" sub-folder

        Args:
        -----
            train_data = dictionary, containing 'X', 'y', 'w' for the training set, where:
                X = ndarray of dim (# training examples, # features)
                y = array of dim (# training examples) with target values
                w = array of dim (# training examples) with event weights
            classification_variables = list of names of variables used for classification
            variable_dict = ordered dict, mapping all the branches from the TTree to their type
        """
        # -- Train:
        logging.getLogger("sklBDT.train").info("Training...")
        classifier = GradientBoostingClassifier(n_estimators=200, min_samples_split=2, max_depth=10, verbose=1)
        classifier.fit(train_data['X'], train_data['y'], sample_weight=train_data['w'])

        # -- Dump output to pickle
        self.ensure_directory("{}/pickle/".format(self.output_directory))
        joblib.dump(classifier, "{}/pickle/sklBDT_clf.pkl".format(self.output_directory), protocol=cPickle.HIGHEST_PROTOCOL)

        self.ensure_directory(os.path.join(self.output_directory, "pickle"))
        joblib.dump(classifier, os.path.join(self.output_directory, "pickle", "sklBDT_clf.pkl"), protocol=cPickle.HIGHEST_PROTOCOL)

    def test(self, data, classification_variables, process):
        """
        Definition:
        -----------
            Testing method for sklBDT; it loads the latest model from the "pickle" sub-folder

        Args:
        -----
            data = dictionary, containing 'X', 'y', 'w' for the set to evaluate performance on, where:
                X = ndarray of dim (# examples, # features)
                y = array of dim (# examples) with target values
                w = array of dim (# examples) with event weights
            process = string to identify whether we are evaluating performance on the train or test set, usually "training" or "testing"
            classification_variables = list of names of variables used for classification

        Returns:
        --------
            yhat = the array of BDT outputs corresponding to the P(signal), of dimensions (n_events)
        """
        logging.getLogger("sklBDT.test").info("Evaluating performance...")

        # -- Load scikit classifier
        classifier = joblib.load("{}/pickle/sklBDT_clf.pkl".format(self.output_directory))

        # -- Get classifier predictions
        yhat = classifier.predict_proba(data['X'])[:, 1]

        # -- Load scikit classifier
        classifier = joblib.load(os.path.join(self.output_directory, "pickle", "sklBDT_clf.pkl"))

        # -- Log classification scores
        logging.getLogger("sklBDT.test").info("{} accuracy = {:.2f}%".format(process, 100 * classifier.score(data['X'], data['y'], sample_weight=data['w'])))
        for output_line in classification_report(data['y'], classifier.predict(data['X']), target_names=["correct", "incorrect"], sample_weight=data['w']).splitlines():
            logging.getLogger("sklBDT.test").info(output_line)

        return yhat
