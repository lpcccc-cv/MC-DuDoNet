import os
from nibabel.viewers import OrthoSlicer3D
from nibabel import nifti1
import nibabel as nib
from matplotlib import pylab as plt
import matplotlib
from PIL import Image
import numpy as np
import cv2
import SimpleITK as sitk

def norm(data):
    data = data.astype(np.float32)
    max = np.max(data)
    min = np.min(data)
    data = (data-min)/(max-min)
    return data*255.

file_path = '/home/lpc/dataset/IXI/IXI-PD'
save_file_path = '/home/lpc/dataset/IXI/MCSR/test/PD'

step = 1
skip = 10
totle_number=0

file_list = sorted(os.listdir(file_path))
with open("/home/lpc/dataset/IXI/T2/test.txt","r") as f:
    for number, name in enumerate(file_list):
        print(name)
        filename_1 = file_path+'/'+name
        img_1 = sitk.ReadImage(filename_1, sitk.sitkInt16)
        space = img_1.GetSpacing()
        img_1 = sitk.GetArrayFromImage(img_1)
        width, height, queue = img_1.shape

        data_1 = norm(img_1)
        for i in range(skip, width-skip, step):

            totle_number = totle_number+1
            img_arr1 = data_1[i, :, :]
            img_arr1 = np.expand_dims(img_arr1, axis=2)

            cv2.imwrite(save_file_path+'/{:08d}.png'.format(totle_number), img_arr1)
            print('Done!'+save_file_path+'/{:08d}.png'.format(totle_number))

