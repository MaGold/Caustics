import scipy
import os
import numpy as np


# Walk through path and get filenames as strings
def get_filenames(path):
    fnames = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        fnames.extend(filenames)
        break
    return fnames


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
    pics = []
    imgshape = []
    count = 0
    for img in fnames:
        if count > num_imgs:
            break
        count += 1
        fullpath = os.path.join(dir_name, img)
        pic = scipy.misc.imread(fullpath)
        if grey and len(pic.shape) == 2:
            pic = pic.reshape(pic.shape[0], pic.shape[1], 1)
        if real:
            subpics = matrix_splitter(pic, shape=(480, 480))
        else:
            subpics = matrix_splitter(pic, shape=(pic.shape[0],
                                                  pic.shape[0]))
        for pic in subpics:
            if grey and pic.shape[2] == 3:
                pic = 0.21*pic[:, :, 0] + 0.72*pic[:, :, 1] + 0.07*pic[:, :, 2]
            elif grey and pic.shape[2] == 1:
                pic = pic.reshape(pic.shape[0], pic.shape[1])
            if pic.shape[0] > img_shape[0] and pic.shape[1] > img_shape[1]:
                pic = scipy.misc.imresize(pic, img_shape)
                imgshape = pic.shape
                pics.append(pic.reshape(-1))
    return pics, imgshape


def get_data(img_shape=(100,100)):
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

    rgb_imgnames1 = sorted(get_filenames(rgb_path1))
    rgb_imgnames2 = sorted(get_filenames(rgb_path2))
    mask_imgnames1 = sorted(get_filenames(mask_path1))
    mask_imgnames2 = sorted(get_filenames(mask_path2))
    real_imgnames = sorted(get_filenames(real_path))

    # Load each pic, resizing, reshaping, etc...
    rgb_pics1, rgb_imgshape = load_imgs(rgb_imgnames1,
                                        rgb_path1,
                                        img_shape=img_shape)
    rgb_pics2, rgb_imgshape = load_imgs(rgb_imgnames2,
                                        rgb_path2,
                                        img_shape=img_shape)
    mask_pics1, mask_imgshape = load_imgs(mask_imgnames1,
                                          mask_path1,
                                          img_shape=img_shape,
                                          grey=True)
    mask_pics2, mask_imgshape = load_imgs(mask_imgnames2,
                                          mask_path2,
                                          img_shape=img_shape,
                                          grey=True)
    real_pics, real_imgshape = load_imgs(real_imgnames,
                                         real_path,
                                         img_shape=img_shape,
                                         real=True)

    rgb_pics = np.concatenate((rgb_pics1, rgb_pics2), axis=0)
    mask_pics = np.concatenate((mask_pics1, mask_pics2), axis=0)

    # Convert the list of images to a numpy array
    X = np.array(rgb_pics)
    Y = np.array(mask_pics)
    # Delete pics to free up memory
    del rgb_pics
    del mask_pics
    # Rescale
    X = X*1.0/np.max(X)
    Y = Y*1.0/np.max(Y)
    real_pics = np.array(real_pics)
    real_pics = real_pics*1.0/np.max(real_pics)
    real_pics = real_pics.astype(np.float32)

    # To keep theano from complaining
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)

    a = np.arange(X.shape[0])
    np.random.shuffle(a)
    indx = a[:200]
    teX = X[indx, :]
    teY = Y[indx, :]
    trX = np.delete(X, indx, 0)
    trY = np.delete(Y, indx, 0)

    teReal = real_pics
    channels = 3
    img_x = rgb_imgshape[0]
    img_y = rgb_imgshape[1]

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

    teReal = real_pics.reshape((real_pics.shape[0],
                                img_x,
                                img_y,
                                channels))
    teReal = np.swapaxes(teReal, 2, 3)
    teReal = np.swapaxes(teReal, 1, 2)
    return trX, trY, teX, teY, teReal, rgb_imgshape


def get_real_data(img_shape=(100, 100)):
    real_path = os.path.join("data", "realfromabove")
    real_imgnames = sorted(get_filenames(real_path))
    print("Loading " + str(len(real_imgnames)) + " images...")
    # Load each pic, resizing, reshaping, etc...
    real_pics, real_imgshape = load_imgs(real_imgnames,
                                         real_path,
                                         img_shape=img_shape,
                                         real=True)
    print("Loading done.")
    # Convert the list of images to a numpy array
    real_pics = np.array(real_pics)
    real_pics = real_pics*1.0/np.max(real_pics)
    real_pics = real_pics.astype(np.float32)
    return real_pics, real_imgshape


def get_synthetic_data():
    synthetic_path = os.path.join("data",
                                  "realfromabove")
    synthetic_path = os.path.join("data",
                                  "datasetv2",
                                  "Underwater_Caustics",
                                  "set2")

    real_imgnames = sorted(get_filenames(synthetic_path))

    # Load each pic, resizing, reshaping, etc...
    real_pics, real_imgshape = load_imgs(real_imgnames,
                                         synthetic_path,
                                         real=False)

    # Convert the list of images to a numpy array
    real_pics = np.array(real_pics)
    real_pics = real_pics*1.0/np.max(real_pics)
    real_pics = real_pics.astype(np.float32)

    test_path = os.path.join("data", "testimgs")
    testimgnames = sorted(get_filenames(test_path))
    testimgs, test_imgshape = load_imgs(testimgnames,
                                        test_path,
                                        real=True)
    testimgs = np.array(testimgs)
    testimgs = testimgs*1.0/np.max(testimgs)
    testimgs = testimgs.astype(np.float32)
    return real_pics, real_imgshape, testimgs
