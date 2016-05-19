# Caustics

Requirements:
- python 3
- theano
- matplotlib

There are two main files to work with directly: `train.py` and `process.py`.
By default, data will be read from the `data` directory. The specific structure of this directory is specified in `load_data.py`. The general model setup is contained in `model.py`. The file `plots.py` contains simple functions for plotting training curves, image reconstructions, and more.

## Training

Training is done using the script `train.py`. The network archicture
is specified in this file. Three directories will created by this script for storing images and network parameters as training progresses. By default, these directories are `training_images`, `validation_images`, and `params`. Training costs and other details will be written to `costs.txt`.

## Processing

Once training is complete and network parameters have been saved in the `params` directory, images can be processed by the network in two ways:

### Python script: process.py

The script in `process.py` can be used to inspect the output of the trained network. Run the script with the directory of the images to be processed as an argument, for example, `python3 process.py data/real_frames/GOPR0085`. The output images will be stored in the directory `caustic_frames`. 

### Batch script: create_output_videos.sh

This script will loop through all the directories in `data/real_frames/`. For each subdirectory, the above python script will be used to

1. Create the corresponding output images
2. Create videos based on the the input and output images

All created files will be stored in the directory `output_videos`