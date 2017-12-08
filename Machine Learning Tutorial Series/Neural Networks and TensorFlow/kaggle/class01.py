import dicom # for reading dicom files
import os # for doing directory operations
import pandas as pd # for some simple data analysis (right now, just to load in the labels data and quickly reference it)

# https://www.kaggle.com/c/data-science-bowl-2017/data

# Section 1

# Change this to wherever you are storing your data:
# IF YOU ARE FOLLOWING ON KAGGLE, YOU CAN ONLY PLAY WITH THE SAMPLE DATA, WHICH IS MUCH SMALLER

data_dir = 'X:/Kaggle_Data/datasciencebowl2017/stage1/'
patients = os.listdir(data_dir)
labels_df = pd.read_csv('stage1_labels.csv', index_col=0)

import matplotlib.pyplot as plt

for patient in patients[:1]:
    label = labels_df.get_value(patient, 'cancer')
    path = data_dir + patient
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))

    #          the first slice
    plt.imshow(slices[0].pixel_array)
    plt.show()