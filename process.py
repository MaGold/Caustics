# ------------------------------------------------------------------------
# This script takes an already trained network and produces 3 things:
# 1) A sequence of greyscale frames indicating caustic probabilities,
#    stored in CAUSTICS_FRAMES_DIR
# 2) A sequence of rgb frames, based on thresholded caustic values,
#    where pixels strongly hypothesized to be caustics are turned white,
#    stored in WHITENED_FRAMES_DIR
# 3) An image showing the filters of the first layer of the network,
#    stored in FILTERS_DIR
#
# Run the script with the directory of the images to be processed
# as an argument:
#    python process.py data/real_frames/GOPR0085
# ------------------------------------------------------------------------

from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import os
import sys
import pickle
import theano
import numpy as np
import model
import load_data
import plots
from imp import reload
# Reload in case local changes were made
reload(model)
reload(load_data)
reload(plots)

# Directories where files are stored or loaded from
CAUSTICS_FRAMES_DIR = "caustic_frames"
WHITENED_FRAMES_DIR = "whitened_frames"
FILTERS_DIR = "filters"
PARAMS_DIR = "params"
TO_PROCESS = sys.argv[1]
os.makedirs(CAUSTICS_FRAMES_DIR, exist_ok=True)
os.makedirs(WHITENED_FRAMES_DIR, exist_ok=True)
os.makedirs(FILTERS_DIR, exist_ok=True)

# Number of pixels in the input and output images
IMG_SHAPE = (400, 400)

theano.config.floatX = 'float32'
srng = RandomStreams()

channels = 3
real = load_data.get_real_data(TO_PROCESS, img_shape=IMG_SHAPE)

img_x = IMG_SHAPE[0]
img_y = IMG_SHAPE[1]

teReal = real.reshape((real.shape[0], img_x, img_y, channels))
teReal = np.swapaxes(teReal, 2, 3)
teReal = np.swapaxes(teReal, 1, 2)

X = T.ftensor4()
Y = T.ftensor4()

# Load the parameters of the previously trained network
filters_file = os.path.join(PARAMS_DIR, "1_filter_params.p")
biases_file = os.path.join(PARAMS_DIR, "1_bias_params.p")
filter_params = pickle.load(open(filters_file, "rb"))
bias_params = pickle.load(open(biases_file, "rb"))
print("Network parameters loaded.")

pred_out = model.model(X, filter_params, bias_params, 0.0, srng)
pred_out_flat = pred_out.flatten(2)

predict = theano.function(inputs=[X],
                          outputs=pred_out, allow_input_downcast=True)

# Use network output to insert white pixels in original real video
outblock = np.zeros((len(teReal), img_x, img_y))
# T = 0.2  # threshold amount
for i in range(teReal.shape[0]):
    samples = teReal[i, :, :, :]
    samples = np.expand_dims(samples, axis=0)
    out = predict(samples)
    out_img = out[0, 0, :, :]
    if i % 50 == 0:
        print(i)
    plots.plot_img(out_img, i, CAUSTICS_FRAMES_DIR)

    # # Threshold the output of the network
    # out[out > T] = 1.0
    # out[out <= T] = 0.0
    # outblock[i, :, :] = out

#     # Create frames of new video with white pixels where caustics were found
#     Rframe = teReal[i, 0, :, :]
#     Gframe = teReal[i, 1, :, :]
#     Bframe = teReal[i, 2, :, :]
#     # Make caustic pixels white:
#     Rframe[outblock[i, :] == 1.0] = 1
#     Gframe[outblock[i, :] == 1.0] = 1
#     Bframe[outblock[i, :] == 1.0] = 1
#     frame = np.dstack((Rframe, Gframe, Bframe))
#     plots.plot_img(frame, i, WHITENED_FRAMES_DIR)


# # Plot filters
# half = int(len(filter_params)/2)
# conv_params = filter_params[:half]
# fs = conv_params[0].get_value()
# plots.plot_rgb_filters(fs, 0, FILTERS_DIR)
