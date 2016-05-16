import numpy as np
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_filters(w, channels, idx, title):
    if channels == 1:
        plot_grey_filters(w, idx, title)
    elif channels == 3:
        plot_rgb_filters(w, idx, title)


def plot_grey_filters(x, idx, title=""):
    num_filters = x.shape[0]
    numrows = 10
    numcols = int(np.ceil(num_filters/10))
    plt.figure(figsize=(numrows, numcols))
    gs = gridspec.GridSpec(numcols, numrows)
    gs.update(wspace=0.1)
    gs.update(hspace=0.0)
    for i in range(num_filters):
        ax = plt.subplot(gs[i])
        w = x[i, :, :, :]
        w = np.swapaxes(w, 0, 1)
        w = np.swapaxes(w, 1, 2)
        ax.imshow(w[:, :, 0], cmap=plt.cm.gist_yarg,
                  interpolation='nearest', aspect='equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.axis('off')
    plt.savefig(os.path.join('filters',
                             title + '_' + str(idx) + '_convcaustics.png'))
    plt.close('all')


def plot_rgb_filters(x, idx, title=""):
    num_filters = x.shape[0]
    numrows = 10
    numcols = int(np.ceil(num_filters/10))
    plt.figure(figsize=(numrows, numcols))
    gs = gridspec.GridSpec(numcols, numrows)
    gs.update(wspace=0.1)
    gs.update(hspace=0.0)
    print("plotting color filters")
    for i in range(num_filters):
        ax = plt.subplot(gs[i])
        w = x[i, :, :, :]
        w = np.swapaxes(w, 0, 1)
        w = np.swapaxes(w, 1, 2)
        ax.imshow(w,
                  cmap=plt.cm.gist_yarg,
                  interpolation='nearest',
                  aspect='equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.axis('off')
    plt.savefig(os.path.join('filters',
                             title + '_' + str(idx) + '_convcaustics.png'))
    print(os.path.join('filters',
                       title + '_' + str(idx) + '_convcaustics.png'))
    plt.close('all')


# Plot a grid with 5 rows and 3 columns
def plot_train(samples, predictions, truths, k, imgshape, dir_name):
    batch_size = samples.shape[0]
    predictions = predictions.reshape(batch_size, imgshape[2], imgshape[3])
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle('Epoch: ' + str(k), size=20)
    gs = gridspec.GridSpec(5, 3)
    for i in range(15):
        ax = plt.subplot(gs[i])
        if i % 3 == 0:
            w = samples[int(i/3), :, :, :]
            w = np.swapaxes(w, 0, 1)
            w = np.swapaxes(w, 1, 2)
        elif i % 3 == 1:
            w = predictions[int(i/3), :, :]
        else:
            w = truths[int(i/3), 0, :, :]
        ax.imshow(w,
                  cmap=plt.cm.gist_yarg,
                  interpolation='nearest',
                  aspect='equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.axis('off')
        if i == 0:
            ax.set_title("Input")
        elif i == 1:
            ax.set_title("Output")
        elif i == 2:
            ax.set_title("Ground truth")
    gs.update(wspace=0)
    plt.savefig(os.path.join(dir_name, str(k) + '_train.png'))
    plt.close('all')


# Plot a grid with 5 rows and 2 columns
def plot_validation(samples, predictions, k, imgshape, dir_name, f_name=""):
    batch_size = samples.shape[0]
    predictions = predictions.reshape(batch_size, imgshape[2], imgshape[3])
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle('Epoch: ' + str(k), size=20)
    gs = gridspec.GridSpec(5, 2)
    for i in range(10):
        ax = plt.subplot(gs[i])
        if i % 2 == 0:
            w = samples[int(i/2), :, :, :]
            w = np.swapaxes(w, 0, 1)
            w = np.swapaxes(w, 1, 2)
        else:
            w = predictions[int(i/2), :, :]
        ax.imshow(w,
                  cmap=plt.cm.gist_yarg,
                  interpolation='nearest',
                  aspect='equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.axis('off')
        if i == 0:
            ax.set_title("Input \n")
        elif i == 1:
            ax.set_title("Output \n")
    gs.update(wspace=0)
    plt.savefig(os.path.join(dir_name,
                             str(k) + '_' + f_name + '_validation.png'))
    plt.close('all')


def plot_training_curve(tr, te, ind):
    # sns.axes_style("darkgrid")
    # fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(111)
    plt.figure(figsize=(12, 8))
    plt.plot(np.log(tr), label='train')
    plt.plot(np.log(te), label='validation')
    plt.legend(prop={'size': 15})
    plt.xlabel('Epoch')
    plt.ylabel('Log(error)')
    plt.savefig('costs.png')
    plt.close('all')


def plot_img(img, indx, dir_name):
    fig = plt.figure(figsize=(4.8, 4.8))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap=plt.cm.gist_yarg,
              interpolation='nearest', aspect='equal')
    plt.savefig(os.path.join(dir_name,
                             str(indx) + '.png'))
    plt.close()
