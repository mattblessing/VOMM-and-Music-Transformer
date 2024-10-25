# VOMM-and-Music-Transformer

This system is a music generator featuring a variable-order Markov model and a Music Transformer, allowing for flexible output generation and diversity control via adjustable hyperparameters. See my report 'report.pdf' for more details.

'main.ipynb' is the entry point for the system and contains code for loading data, preprocessing, training the models, generating output and carrying out evaluation. To generate output from the models, import the dependencies and scroll down to the generation section - the pretrained models can be loaded and output can be generated and listened to.

The 'data' directory contains the pretrained models and samples used in the evaluation, so no training is necessary to generate outputs. However, the MAESTRO V3.0.0 MIDI dataset can be downloaded [here](https://magenta.tensorflow.org/datasets/maestro#v300) and preprocessed for (further) training.

The list of required third-party modules are in 'requirements.txt' and these should be installed prior to use.

Development was done in Python 3.11.
