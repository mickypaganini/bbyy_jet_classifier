# bbyy_jet_classifier
This package is used by the ATLAS HH->bbyy search. <br/>
It defines a classifier to choose which non-b-tagged jet to pair with the b-tagged jet in 1-tag events. <br/>
This code uses the `skTMVA` class from https://github.com/yuraic/koza4ok to convert ```scikit-learn``` output into `ROOT TMVA` xml input.

## Dependencies
This package relies on two different Machine Learning libraries to train BDTs, and the user can control which one to use. One is [`scikit-learn`](http://scikit-learn.org/stable/install.html "Scikit-learn Installation"); the other is [`TMVA`](http://tmva.sourceforge.net/), a CERN specific data science package included in [`ROOT`](https://root.cern.ch/). <br/>
Both `ROOT` and `scikit-learn` are required even if the user decides to only use one of the two for the ML portion of the project. `ROOT` is required to open the `.root` input files and `scikit-learn` is used in different capacities during the data pre-processing stages.
Other necessary libraries include:
* [joblib](https://pythonhosted.org/joblib/installing.html)
* [matplotlib](http://matplotlib.org/faq/installing_faq.html)
* [numpy](http://docs.scipy.org/doc/numpy-1.10.0/user/install.html)
* [rootpy](http://www.rootpy.org/install.html)
* [root_numpy](https://rootpy.github.io/root_numpy/install.html)

For updated requirements check the requirements.txt file.

## Inputs
The most recent set of input TTrees are in:
```
/afs/cern.ch/user/j/jrobinso/work/public/ML_inputs
```

## Structure
There are two main script to execute: `run_classifier.py` and `evaluate_event_performance.py`. These need to be executed sequentially whenever the output of the former looks satisfactory. <br/>
One could also use the script `bbyy_jet_classifier/process_data.py` first to turn root files into pickle files of pre-processed data. This step is not necessary, because `run_classifier.py` will automatically invoke `bbyy_jet_classifier/process_data.py` if pre-processed data is not found. <br/>

### Usage of `bbyy_jet_classifier/process_data.py`:
This pre-processing step may be useful to ensure that the data is being processed correctly without corruption or unwanted results. It also allows us to re-use the data and make sure that samples that were previously used for training will always be separate from those used for testing. 
```
usage: process_data.py [-h] --input INPUT [INPUT ...] [--tree TREE]
                       [--ftrain FTRAIN]

Pre-process data for 2nd jet classifier. Save to pkl.

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT [INPUT ...]
                        List of input root file paths
  --tree TREE           Name of the tree in the ntuples. Default: events_1tag
  --ftrain FTRAIN       Fraction of events to allocate for training. Default:
                        0.7.
```
It will process a series of root files, shuffle them, split them into training and testing samples according to the fraction specified using `--ftrain`, scale them, put them in convenient `X`, `y`, `w` format plus extra variables used for performance evaluation, store them in dictionaries, and save them out to pickle files with the following name format: `{fname}---{tree}---{frac}---train.pkl`, `{fname}---{tree}---{frac}---test.pkl`.

### Usage of `run_classifier.py`:
This is the main script. It is composed of different parts that handle data processing, plotting, training and/or testing. <br/>
Training and testing have been divided such that the two tasks can be performed indipendently on separate runs of this code. To specify the mode, pass it as a command to the command line:
```
usage: run_classifier.py [-h] {train,test} ...

Run ML classifiers for 2nd jet selection

optional arguments:
  -h, --help    show this help message and exit

actions:
  {train,test}
    train       train BDT for jet-pair classification
    test        test BDT for jet-pair classification
```
The `train` mode takes the following arguments:
```
usage: run_classifier.py train [-h] --input INPUT [INPUT ...] [--tree TREE]
                               [--strategy STRATEGY [STRATEGY ...]]
                               [--exclude VARIABLE_NAME [VARIABLE_NAME ...]]
                               [--max_events MAX_EVENTS] [--output OUTPUT]
                               --training_id TRAINING_ID [--grid_search]

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT [INPUT ...]
                        List of original ROOT input file paths
  --tree TREE           Name of the tree in the ntuples. Default: events_1tag
  --strategy STRATEGY [STRATEGY ...]
                        Type of BDT to use. Options are: RootTMVA, sklBDT.
                        Default: both
  --exclude VARIABLE_NAME [VARIABLE_NAME ...]
                        List of variables that are present in the tree but
                        should not be used by the classifier
  --max_events MAX_EVENTS
                        Maximum number of events to use (for debugging).
                        Default: all
  --output OUTPUT       Output directory. Default: output
  --training_id TRAINING_ID
                        String to identify training in the future or to select
                        which pretrained model to test
  --grid_search         Pass this flag to run a grid search to determine BDT
                        parameters
```
The `test` mode takes the following arguments:
```
usage: run_classifier.py test [-h] --input INPUT [INPUT ...] [--tree TREE]
                              [--strategy STRATEGY [STRATEGY ...]]
                              [--exclude VARIABLE_NAME [VARIABLE_NAME ...]]
                              [--output OUTPUT] --training_id TRAINING_ID
                              [--max_events MAX_EVENTS]

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT [INPUT ...]
                        List of original ROOT input file paths
  --tree TREE           Name of the tree in the ntuples. Default: events_1tag
  --strategy STRATEGY [STRATEGY ...]
                        Type of BDT to use. Options are: RootTMVA, sklBDT.
                        Default: both
  --exclude VARIABLE_NAME [VARIABLE_NAME ...]
                        List of variables that are present in the tree but
                        should not be used by the classifier
  --output OUTPUT       Output directory. Default: output
  --training_id TRAINING_ID
                        String to identify training in the future or to select
                        which pretrained model to test
```

### Usage of `evaluate_event_performance.py`:
Event level performance evaluation quantified in terms of Asimov significance. <br/>
It compares the performance of three old strategies (`mHmatch`, `pThigh`, `pTjb`) with that of the BDT. The BDT performance is evaluated after excluding events in which the highest BDT score is < threshold. For many threshold values, the performance can be computed in paralled. <br/>
It outputs a plot and a pickled dictionary.
```
usage: evaluate_event_performance.py [-h] [--strategy STRATEGY]
                                     [--category CATEGORY]
                                     [--intervals INTERVALS]

Check event level performance

optional arguments:
  -h, --help            show this help message and exit
  --strategy STRATEGY   Strategy to evaluate. Options are: root_tmva, skl_BDT.
                        Default: skl_BDT
  --category CATEGORY   Trained classifier to use for event-level evaluation.
                        Examples are: low_mass, high_mass. Default: low_mass
  --intervals INTERVALS
                        Number of threshold values to test. Default: 21
```

# Examples: 
### Training on SM inputs
This will train two BDTs (one with `TMVA`, one with `scikit-learn`) on 1000 events (for debugging purposes) from the Standard Model input files, without usin the Delta_phi_jb variable.
```
python run_classifier.py train --input inputs/SM_bkg_photon_jet.root inputs/SM_hh.root --exclude Delta_phi_jb --strategy RootTMVA sklBDT --max_events 1000
```

### Testing previously trained classifier on different individual BSM inputs
This will only test the performance of a previously trained BDT (the location of which is indicated by `--training_id`) on the input files specified after `--input`. In this case, these examples show how to test the performance of a scikit-learn BDT trained on merged SM samples on various BSM samples.
```
python run_classifier.py test --input inputs/X275_hh.root --exclude Delta_phi_jb --strategy sklBDT --training_id SM_merged

python run_classifier.py test --input inputs/X300_hh.root --exclude Delta_phi_jb --strategy sklBDT --training_id SM_merged

python run_classifier.py test --input inputs/X325_hh.root --exclude Delta_phi_jb --strategy sklBDT --training_id SM_merged

python run_classifier.py test --input inputs/X350_hh.root --exclude Delta_phi_jb --strategy sklBDT --training_id SM_merged

python run_classifier.py test --input inputs/X400_hh.root --exclude Delta_phi_jb --strategy sklBDT --training_id SM_merged
```

### Training and testing on all inputs
By default, 70% of the input files will be used for training, 30% for testing.
```
python run_classifier.py train --input inputs/*root --exclude Delta_phi_jb --training_id all
python run_classifier.py test --input inputs/*root --exclude Delta_phi_jb --training_id all
```

### Training on low-mass inputs
The signal samples in the mass range between 275 and 350 GeV are commonly identified as the "low mass" samples. A classifier trained on those will then be identified by that tag `low_mass` and placed in an homonymous folder. 
```
python run_classifier.py train --input inputs/SM_bkg_photon_jet.root inputs/X275_hh.root inputs/X300_hh.root inputs/X325_hh.root inputs/X350_hh.root --exclude Delta_phi_jb --training_id low_mass
```

### Training on high-mass inputs
Conversely, the Standard Model signal samples and the BSM sample with resonant mass of 400 GeV are commonly identified as the "high mass samples. A classifier trained on those will then be identified by that tag `high_mass` and placed in an homonymous folder. 
```
python run_classifier.py train --input inputs/SM_bkg_photon_jet.root inputs/X400_hh.root inputs/SM_hh.root --exclude Delta_phi_jb --training_id high_mass
```

### Evaluate event-level performance
`run_classifier.py` only handles the pipeline to the individual jet pair classification stage. A final step is to evaluate the event-level performance of the tagger, by selecting in each event the jet pair that has the highest BDT score and checking how often that corresponds to the correct pair. The event-level performance is then compared with that of other methods that were previosuly tested in the analysis.
```
python evaluate_event_performance.py --category low_mass --strategy root_tmva --intervals 21
```
