# Caustics

Requirements:
python3
theano

There are two main files to work with directly: `train.py` and `process.py`.

## Training

Training is done using the script `train.py`. The network archicture
is specified in this file. Three directories will created by this script for storing images and network parameters as the network progresses. By default, these directories are `training_images`, `validation_images`, and `params`.

## Processing

Once training is complete and network parameters have been saved in the `params` directory, the script in `process.py` can be used to inspect the trained network.

