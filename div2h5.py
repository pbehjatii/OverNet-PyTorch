import os
import glob
import h5py
import scipy.misc as misc
import numpy as np
import cv2
dataset_dir = "datasets/DIV2K/"
dataset_type = "train"

f = h5py.File("datasets/DIV2K_{}.h5".format(dataset_type), "w")

dt = h5py.special_dtype(vlen=np.dtype('uint8'))

for subdir in ["HR", "X2", "X3", "X4"]:
    if subdir in ["HR"]:
        im_paths = glob.glob(os.path.join(dataset_dir,
                                          "DIV2K_{}_HR".format(dataset_type),
                                          "*.png"))

    else:
        im_paths = glob.glob(os.path.join(dataset_dir,
                                          "DIV2K_{}_LR_bicubic".format(dataset_type),
                                          subdir, "*.png"))
    im_paths.sort()
    grp = f.create_group(subdir)

    for i, path in enumerate(im_paths):
        im = cv2.imread(path)
        print(path)
        grp.create_dataset(str(i), data=im)
