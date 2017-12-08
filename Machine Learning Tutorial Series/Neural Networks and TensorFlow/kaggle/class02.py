import cv2
import numpy as np
import pandas as pd
import os
import dicom
import matplotlib.pyplot as plt

import math

# Processing and viewing our Data

def chunks(l, n):
    # Credit: Ned Batchelder
    # Link: http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def mean(l):
    return sum(l) / len(l)

IMG_PX_SIZE = 150
HM_SLICES = 20


data_dir = 'X:/Kaggle_Data/datasciencebowl2017/stage1/'
patients = os.listdir(data_dir)
labels_df = pd.read_csv('stage1_labels.csv', index_col=0)
for patient in patients[:1]:
    label = labels_df.get_value(patient, 'cancer')
    path = data_dir + patient
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
    new_slices = []

    slices = [cv2.resize(np.array(each_slice.pixel_array), (IMG_PX_SIZE, IMG_PX_SIZE)) for each_slice in slices]

    chunk_sizes = math.ceil(len(slices) / HM_SLICES)
    for slice_chunk in chunks(slices, chunk_sizes):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)

    if len(new_slices) == HM_SLICES - 1:
        new_slices.append(new_slices[-1])

    if len(new_slices) == HM_SLICES - 2:
        new_slices.append(new_slices[-1])
        new_slices.append(new_slices[-1])

    if len(new_slices) == HM_SLICES + 2:
        new_val = list(map(mean, zip(*[new_slices[HM_SLICES - 1], new_slices[HM_SLICES], ])))
        del new_slices[HM_SLICES]
        new_slices[HM_SLICES - 1] = new_val

    if len(new_slices) == HM_SLICES + 1:
        new_val = list(map(mean, zip(*[new_slices[HM_SLICES - 1], new_slices[HM_SLICES], ])))
        del new_slices[HM_SLICES]
        new_slices[HM_SLICES - 1] = new_val

    fig = plt.figure()
    for num, each_slice in enumerate(new_slices):
        y = fig.add_subplot(4, 5, num + 1)
        y.imshow(each_slice, cmap='gray')
    plt.show()