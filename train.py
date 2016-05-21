# ------------------------------------------------------------------------
# This script is used to train a convolutional neural network.
# Training can be monitored by looking at:
#     TRAIN_DIR: sample training data is plotted and saved in this directory
#     SYNTHETIC_VAL_DIR: sample synthetic data is plotted and
#                        saved in this directory
#     REAL_VAL_DIR: sample real images plotted and saved in this directory
#
# The parameters of the network are occasionally saved to files in PARAMS_DIR
# ------------------------------------------------------------------------

from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import os
import pickle
import theano
import numpy as np
import time
# Local modules
import model
import load_data
import plots
from imp import reload
# Reload these in case local changes were made
reload(model)
reload(load_data)
reload(plots)


# Helper function for writing output to file
def write(str):
    f = open("costs.txt", 'a')
    f.write(str)
    f.close()

# Directory where files will be saved
PARAMS_DIR = "params"
TRAIN_DIR = "training_images"
SYNTHETIC_VAL_DIR = "synthetic_validation_images"
REAL_VAL_DIR = "real_validation_images"
os.makedirs(PARAMS_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(SYNTHETIC_VAL_DIR, exist_ok=True)
os.makedirs(REAL_VAL_DIR, exist_ok=True)

# Number of pixels in the input and output images
IMG_SHAPE = (400, 400)

theano.config.floatX = 'float32'
srng = RandomStreams()
f = open("costs.txt", 'w')
f.write("Starting...\n")
f.close()

print("Loading image data...")
write("Loading image data...")

channels = 3
trX, trY, teX, teY, teReal = load_data.get_data(img_shape=IMG_SHAPE)
img_x = IMG_SHAPE[0]
img_y = IMG_SHAPE[1]

s1 = str(trX.shape[0]) + " synthetic training images.\n"
s2 = str(teX.shape[0]) + " synthetic validation images.\n"
s3 = str(teReal.shape[0]) + " real images.\n"
print(s1 + s2 + s3)
write(s1 + s2 + s3)

X = T.ftensor4()
Y = T.ftensor4()

# Network architecture
f1 = (5, channels, 3, 3)  # 5 filters of shape 3 x 3
filters = [f1]

# More layers can be added.
# The following would yield a network with 3 convolutional layers,
# followed by 3 deconvolutional layers
# f1 = (5, channels, 3, 3)
# f2 = (20, f1[0], 3, 3)
# f3 = (10, f2[0], 3, 3)
# filters = [f1, f2, f3]

filter_params, bias_params = model.get_params(img_x, filters)

# Model with dropout for training
# Note: dropout is implemented but not used
noise_out = model.model(X, filter_params, bias_params, 0.0, srng)
noise_out_flat = noise_out.flatten(2)

# Model without dropout for validating
pred_out = model.model(X, filter_params, bias_params, 0.0, srng)
pred_out_flat = pred_out.flatten(2)

flat_y = Y.flatten(2)

# Mean Squared Error
L_noise = T.sum((flat_y - noise_out_flat)**2, axis=1)

# Alternatively, the following can be used for cross-entropy
# noise_out = T.clip(noise_out, 10**(-6), 1-10**(-6))
# L = -T.sum(flat_y * T.log(noise_out) +
#            (1-flat_y) * T.log(1-noise_out), axis=1)
noise_cost = L_noise.mean()

L_clean = T.sum((flat_y - pred_out_flat)**2, axis=1)
# clean_out = T.clip(pred_out_flat, 10**(-6), 1-10**(-6))
# L_clean = -T.sum(flat_y * T.log(clean_out) +
#            (1-flat_y) * T.log(1-clean_out), axis=1)
clean_cost = L_clean.mean()

params = filter_params + bias_params
updates = model.RMSprop(noise_cost, params, lr=0.001)

train = theano.function(inputs=[X, Y],
                        outputs=noise_cost,
                        updates=updates,
                        allow_input_downcast=True)

predict = theano.function(inputs=[X],
                          outputs=pred_out,
                          allow_input_downcast=True)

error = theano.function(inputs=[X, Y],
                        outputs=clean_cost,
                        allow_input_downcast=True)

# ------------------------------------------------
# Training loop begins
# ------------------------------------------------

NUM_EPOCHS = 200
BATCH_SIZE = 32

# 5 indices for sample predictions
tr_indx = np.random.randint(0, trX.shape[0], size=5)
te_indx = np.random.randint(0, teX.shape[0], size=5)
re_indx = range(5)

# Store the training and validation costs
tr_errs = []
te_errs = []

start_time = time.time()
for i in range(NUM_EPOCHS):
    # Occasionally save current network parameters to files:
    if i % 1 == 0:
        filters_file = os.path.join(PARAMS_DIR,
                                    str(i) + "_filter_params.p",)
        pickle.dump(filter_params, open(filters_file, "wb"))
        biases_file = os.path.join(PARAMS_DIR,
                                   str(i) + "_bias_params.p",)
        pickle.dump(bias_params, open(biases_file, "wb"))
    # Get training costs:
    er = 0
    c = 0
    for start, end in zip(range(0, len(trX), BATCH_SIZE),
                          range(BATCH_SIZE, len(trX), BATCH_SIZE)):
        er += error(trX[start:end, :], trY[start:end, :])
        c += 1
    tr_errs.append(er/c)

    # Get validation costs:
    er = 0
    c = 0
    for start, end in zip(range(0, len(teX), BATCH_SIZE),
                          range(BATCH_SIZE, len(teX), BATCH_SIZE)):
        er += error(teX[start:end, :], teY[start:end, :])
        c += 1
    te_errs.append(er/c)

    # Update training curve:
    plots.plot_training_curve(tr_errs, te_errs, i)

    # Plot some training predictions:
    samples = trX[tr_indx, :]
    out = predict(samples)
    truths = trY[tr_indx, :]
    plots.plot_train(samples, out, truths, i,
                     [5, 3, img_x, img_y], TRAIN_DIR, "train")

    # Plot some validation predictions for real and synthetic images:
    samples = teX[te_indx, :]
    out = predict(samples)
    truths = teY[te_indx, :]
    plots.plot_train(samples, out, truths, i,
                     [5, 3, img_x, img_y], SYNTHETIC_VAL_DIR, "synthetic")
    samples = teReal[re_indx, :]
    out = predict(samples)
    plots.plot_validation(samples, out, i,
                          [5, 3, img_x, img_y], REAL_VAL_DIR, "real")

    # Actual training step:
    # rindx = np.arange(trX.shape[0])
    # np.random.shuffle(rindx)
    for start, end in zip(range(0, len(trX), BATCH_SIZE),
                          range(BATCH_SIZE, len(trX), BATCH_SIZE)):
        # r = rindx[start:end]
        # batch_cost = train(trX[r, :], trY[r, :])
        batch_cost = train(trX[start:end, :], trY[start:end, :])
        write("Epoch: " + str(i) +
              ", index: " + str(start) +
              ", cost: " + str(batch_cost) + "\n")

    diff = time.time() - start_time
    minutes = diff / 60.0
    write("Epoch: " + str(i) + "\n")
    write(str(minutes) + " minutes\n")
    print("Epoch: " + str(i) + "\n")
    print(str(minutes) + " minutes\n")
