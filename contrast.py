# Read the images in a directory, adjust contrast, and save as a new file

import load_data
import os
from PIL import Image, ImageEnhance
import numpy as np


# Load images in given directory
def adjust_contrast(fnames, read_dir, write_dir, amount=1.0):
    count = 0
    for img in fnames:
        count += 1
        fullpath = os.path.join(read_dir, img)
        img_name = img
        img = Image.open(fullpath)
        contraster = ImageEnhance.Contrast(img)
        ccc = contraster.enhance(amount)
        fname = os.path.join(write_dir, img_name)
        ccc.save(fname)
        img = np.array(ccc)
    return


AMOUNT = 3.0
read_dir = os.path.join("data", "real_frames", "GOPR0380")

write_dir = os.path.join("data", "real_frames", "GOPR0380_" + str(AMOUNT))


os.makedirs(write_dir, exist_ok=True)
img_names = sorted(load_data.get_filenames(read_dir))
adjust_contrast(img_names, read_dir, write_dir, amount=AMOUNT)
