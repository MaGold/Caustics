import scipy
import os
import numpy as np


# Walk through path and get filenames as strings
def get_filenames(path):
    f_names = []
    for (dir_path, dir_names, file_names) in os.walk(path):
        f_names.extend(file_names)
        break
    return f_names


# Extracts submatrices of M having shape `shape`
def matrix_splitter(M, shape):
    row = shape[0]
    col = shape[1]
    i = 0
    j = 0
    submatrices = []
    while i + row <= M.shape[0]:
        while j + col <= M.shape[1]:
            submatrices.append(M[i:i+row, j:j+row, :])
            j += col
        i += row
        j = 0
    return submatrices


# Load images in given directory
def load_imgs(fnames, dir_name, img_shape,
              num_imgs=99999, grey=False, real=False):
    imgs = []
    count = 0
    for img in fnames:
        if count > num_imgs:
            break
        count += 1
        fullpath = os.path.join(dir_name, img)
        img = scipy.misc.imread(fullpath)
        if grey and len(img.shape) == 2:
            img = img.reshape(img.shape[0], img.shape[1], 1)
        if real:
            sub_imgs = matrix_splitter(img, shape=(480, 480))
        else:
            sub_imgs = matrix_splitter(img, shape=(img.shape[0],
                                                   img.shape[0]))
        for img in sub_imgs:
            if grey and img.shape[2] == 3:
                img = 0.21*img[:, :, 0] + 0.72*img[:, :, 1] + 0.07*img[:, :, 2]
            elif grey and img.shape[2] == 1:
                img = img.reshape(img.shape[0], img.shape[1])
            if img.shape[0] > img_shape[0] and img.shape[1] > img_shape[1]:
                img = scipy.misc.imresize(img, img_shape)
                imgs.append(img.reshape(-1))
    return imgs


def get_data(img_shape=(100, 100)):
    rgb_path1 = os.path.join("data",
                             "datasetv2",
                             "Underwater_Caustics",
                             "set1")
    rgb_path2 = os.path.join("data",
                             "datasetv2",
                             "Underwater_Caustics",
                             "set2")
    mask_path1 = os.path.join("data",
                              "datasetv2",
                              "Mask",
                              "set1")
    mask_path2 = os.path.join("data",
                              "datasetv2",
                              "Mask",
                              "set2_actual_thresholded_mask")
    real_path = os.path.join("data",
                             "real")

    rgb_img_names1 = sorted(get_filenames(rgb_path1))
    rgb_img_names2 = sorted(get_filenames(rgb_path2))
    mask_img_names1 = sorted(get_filenames(mask_path1))
    mask_img_names2 = sorted(get_filenames(mask_path2))
    real_img_names = sorted(get_filenames(real_path))

    # Load each img, resizing, reshaping, etc...
    rgb_imgs1 = load_imgs(rgb_img_names1, rgb_path1, img_shape=img_shape)
    rgb_imgs2 = load_imgs(rgb_img_names2, rgb_path2, img_shape=img_shape)
    mask_imgs1 = load_imgs(mask_img_names1, mask_path1, img_shape=img_shape,
                           grey=True)
    mask_imgs2 = load_imgs(mask_img_names2, mask_path2, img_shape=img_shape,
                           grey=True)
    real_imgs = load_imgs(real_img_names, real_path, img_shape=img_shape,
                          real=True)
    rgb_imgs = np.concatenate((rgb_imgs1, rgb_imgs2), axis=0)
    mask_imgs = np.concatenate((mask_imgs1, mask_imgs2), axis=0)

    # Convert the list of images to a numpy array
    X = np.array(rgb_imgs)
    Y = np.array(mask_imgs)

    # Delete imgs to free up memory
    del rgb_imgs
    del mask_imgs

    # Rescale
    X = X*1.0/np.max(X)
    Y = Y*1.0/np.max(Y)
    real_imgs = np.array(real_imgs)
    real_imgs = real_imgs*1.0/np.max(real_imgs)
    real_imgs = real_imgs.astype(np.float32)

    # To keep theano from complaining
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)

    # Shuffle the data and set aside 200 synthetic validation images
    a = np.arange(X.shape[0])
    np.random.shuffle(a)
    indx = a[:200]
    teX = X[indx, :]
    teY = Y[indx, :]
    trX = np.delete(X, indx, 0)
    trY = np.delete(Y, indx, 0)

    teReal = real_imgs
    channels = 3
    img_x = img_shape[0]
    img_y = img_shape[1]

    # Reshape data into the format theano wants it for convolutions
    trX = trX.reshape((trX.shape[0], img_x, img_y, channels))
    trX = np.swapaxes(trX, 2, 3)
    trX = np.swapaxes(trX, 1, 2)

    trY = trY.reshape((trY.shape[0], img_x, img_y, 1))
    trY = np.swapaxes(trY, 2, 3)
    trY = np.swapaxes(trY, 1, 2)

    teX = teX.reshape((teX.shape[0], img_x, img_y, channels))
    teX = np.swapaxes(teX, 2, 3)
    teX = np.swapaxes(teX, 1, 2)

    teY = teY.reshape((teY.shape[0], img_x, img_y, 1))
    teY = np.swapaxes(teY, 2, 3)
    teY = np.swapaxes(teY, 1, 2)

    teReal = real_imgs.reshape((real_imgs.shape[0], img_x, img_y, channels))
    teReal = np.swapaxes(teReal, 2, 3)
    teReal = np.swapaxes(teReal, 1, 2)
    return trX, trY, teX, teY, teReal


def get_real_data(img_shape=(100, 100)):
    real_path = os.path.join("data", "realfromabove")
    real_img_names = sorted(get_filenames(real_path))
    print("Loading " + str(len(real_img_names)) + " images...")
    # Load each img, resizing, reshaping, etc...
    real_imgs = load_imgs(real_img_names,
                          real_path,
                          img_shape=img_shape,
                          real=True)
    print("Loading done.")
    # Convert the list of images to a numpy array
    real_imgs = np.array(real_imgs)
    real_imgs = real_imgs*1.0/np.max(real_imgs)
    real_imgs = real_imgs.astype(np.float32)
    return real_imgs
