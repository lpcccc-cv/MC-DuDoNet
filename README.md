
# Joint Under-Sampling Pattern and Dual-Domain Reconstruction for Accelerating Multi-Contrast MRI

### Authors: Pengcheng Lei, Le Hu, Faming Fang and Guixu Zhang

## 1、Environment
Python 3.8, Pytorch1.7.1, Cuda 11.0.
```
# Compile DCNv2:
cd $ROOT/models/modules/DCNv2
sh make.sh
```
For more implementation details about DCN, please see [[DCNv2]](https://github.com/lucasjinreal/DCNv2_latest).

## 2、 Datasets
### 2.1. Parpare Datasets for IXI and BrainTS:
The IXI and BraTS2018 can be downloaded at:
 [[IXI dataset]](https://brain-development.org/ixi-dataset/) and  [[BrainTS dataset]](http://www.braintumorsegmentation.org/).    
(1) The original data are _**.nii**_ data. Split your data set into training sets, validation sets, and test sets;  
(2) Read _**.nii**_ data and save these slices as **_.png_** images into two different folders as:
```bash
python data/read_nii_to_img.py

[T1 folder:]
000001.png,  000002.png,  000003.png,  000003.png ...
[T2 folder:]
000001.png,  000002.png,  000003.png,  000003.png ...
# Note that the images in the T1 and T2 folders correspond one to one. The undersampled target images will be automatically generated in the training phase.
```
### 2.2. Parpare Datasets for Fastmri:
The original Fastmri dataset can be downloaded at: [[Fastmri dataset]](https://fastmri.med.nyu.edu/).    
(1) For the paired Fastmri data (PD and FSPD), we follow the data preparation process of MINet and MTrans. For more details, please see [[MINet]](https://github.com/chunmeifeng/MINet) and [[MTrans]](https://github.com/chunmeifeng/MTrans).   
(2) In our code, you can prepare fastmri dataset using **[data/fastmri_dataset.py]**.
## 3、 Model training for joint optimization: 
### 3.1. Joint training
Set your data set path and training parameters in **[configs/joint_optimization.yaml]**, then run 
```bash
sh train_joint.sh
```
### 3.2. Binarize the soft mask to hard mask. 
```bash
CUDA_VISIBLE_DEVICES=0 python test_loupe_mask.py
```
### 3.3. Reconstruction network fine-tuning
Set your data set path and training parameters in **[configs/only_reconstruction.yaml]**. Set your learned mask path in the dataset file and then run 
```bash
sh train_rec.sh
```

## 4、 Model training for conventional MCMRI SR or Reconstruction (Predefined masks):

Set your data set path, mask path and training parameters in **[configs/only_reconstruction.yaml]** and the dataset file, then run 
```bash
sh train_rec.sh
```

## 5、 Model testing:

Modify the test configurations in Python file **[test_psnr.py]**. Then run:
```bash
CUDA_VISIBLE_DEVICES=0 python test_PSNR.py
```

## Acknowledgement
Our codes are built based on [LOUPE](https://github.com/cagladbahadir/LOUPE/) and [BasicSR](https://github.com/XPixelGroup/BasicSR), thank them for releasing their codes!




