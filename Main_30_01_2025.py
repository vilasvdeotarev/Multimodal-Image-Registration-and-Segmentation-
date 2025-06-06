import numpy as np
import os
import cv2 as cv
from numpy import matlib
import nrrd
from Global_vars import Global_vars
from KOA import KOA
from LEO import LEO
from Model_A_3D_TRSNet import Model_A_3D_TRSNet
from Objective_Function import objfun_Segmentation
from PROPOSED import PROPOSED
from Plot_Results import *
from WOA import WOA
from ZOA import ZOA

# Read the dataset
an = 0
if an == 1:
    Dataset = './Datasets/Dataset_1/'
    CT_Images = []
    MR_Images = []
    CT = Dataset + 'CT'
    MR = Dataset + 'MR'
    MR_exhaled_inhaled_dir = os.listdir(MR)
    CT_exhaled_inhaled_dir = os.listdir(CT)
    for j in range(len(MR_exhaled_inhaled_dir)):
        CT_exhaled_inhaled = CT + '/' + CT_exhaled_inhaled_dir[j]
        MR_exhaled_inhaled = MR + '/' + MR_exhaled_inhaled_dir[j]
        CT_nrrd_dir = os.listdir(CT_exhaled_inhaled)
        MR_nrrd_dir = os.listdir(MR_exhaled_inhaled)
        for k in range(len(CT_nrrd_dir)):
            try:
                CT_nrrd_files = CT_exhaled_inhaled + '/' + CT_nrrd_dir[k]
                MR_nrrd_files = MR_exhaled_inhaled + '/' + MR_nrrd_dir[k]
                MR_data, header_mri = nrrd.read(MR_nrrd_files)
                CT_data, header_ct = nrrd.read(CT_nrrd_files)
                CT_Pateint = []
                MR_Pateint = []
                min_slices = min(MR_data.shape[2], CT_data.shape[2])
                for i in range(min_slices):  # slices  MR_data.shape[2]
                    print('MR', j, len(MR_exhaled_inhaled_dir), k, len(MR_nrrd_dir), i, MR_data.shape[2])
                    mri_slice_img = MR_data[:, :, i]  # Extract 2D slice
                    mri_slice_img = (mri_slice_img - np.min(mri_slice_img)) / (
                            np.max(mri_slice_img) - np.min(mri_slice_img)) * 255
                    mri_slice_img = mri_slice_img.astype(np.uint8)
                    mri_slice_img = cv.resize(mri_slice_img, (512, 512))

                    cv.imwrite('./Images/MR/MR_' + str(MR_exhaled_inhaled_dir[j]) + '_' + str(k + 1) + '_image_' + str(
                        i + 1) + '.png', mri_slice_img)
                    MR_Images.append(mri_slice_img)

                for l in range(min_slices):  # slices  CT_data.shape[2]
                    print('CT', j, len(CT_exhaled_inhaled_dir), k, len(CT_nrrd_dir), l, CT_data.shape[2])
                    ct_slice_img = CT_data[:, :, l]  # Extract 2D slice
                    ct_slice_img = (ct_slice_img - np.min(ct_slice_img)) / (
                            np.max(ct_slice_img) - np.min(ct_slice_img)) * 255
                    ct_slice_img = ct_slice_img.astype(np.uint8)
                    ct_slice_img = cv.resize(ct_slice_img, (512, 512))

                    cv.imwrite('./Images/CT/CT_' + str(CT_exhaled_inhaled_dir[j]) + '_' + str(k + 1) + '_image_'
                               + str(l + 1) + '.png', ct_slice_img)
                    CT_Images.append(ct_slice_img)

            except:
                continue

    MR_Images = np.asarray(MR_Images)
    CT_Images = np.asarray(CT_Images)

    np.save('CT_Images.npy', CT_Images)
    np.save('MR_Images.npy', MR_Images)


# Generate Target
an = 0
if an == 1:
    Tar = []
    Ground_Truth = np.load('Ground_Truth.npy', allow_pickle=True)
    for i in range(len(Ground_Truth)):
        print(i, len(Ground_Truth))
        image = Ground_Truth[i]
        result = image.astype('uint8')
        uniq = np.unique(result)
        if len(uniq) > 1:
            Tar.append(1)
        else:
            Tar.append(0)
    Tar = (np.asarray(Tar).reshape(-1, 1)).astype('int')
    np.save('Target.npy', Tar)

# optimization for Segmentation
an = 0
if an == 1:
    Feat = np.load('CT_Images.npy', allow_pickle=True)
    MR_Images = np.load('MR_Images.npy', allow_pickle=True)
    Global_vars.Feat = Feat
    Global_vars.Target = MR_Images
    Npop = 10
    Chlen = 3  # Hidden Neuron Count, epoch, steps per epoch in A-ViTMUnet
    xmin = matlib.repmat(np.asarray([5, 5, 300]), Npop, 1)
    xmax = matlib.repmat(np.asarray([255, 50, 1000]), Npop, 1)
    fname = objfun_Segmentation
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 50

    print("ZOA...")
    [bestfit1, fitness1, bestsol1, time1] = ZOA(initsol, fname, xmin, xmax, Max_iter)

    print("WOA...")
    [bestfit2, fitness2, bestsol2, time2] = WOA(initsol, fname, xmin, xmax, Max_iter)

    print("KOA...")
    [bestfit3, fitness3, bestsol3, time3] = KOA(initsol, fname, xmin, xmax, Max_iter)

    print("LEO...")
    [bestfit4, fitness4, bestsol4, time4] = LEO(initsol, fname, xmin, xmax, Max_iter)

    print("PROPOSED...")
    [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)

    BestSol_CLS = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(), bestsol5.squeeze()]
    fitness = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(), fitness5.squeeze()]

    np.save('Fitness.npy', np.asarray(fitness))
    np.save('BestSol_CLS.npy', np.asarray(BestSol_CLS))

# Image Registration and segmentation
an = 0
if an == 1:
    image1 = np.load('CT_Images.npy', allow_pickle=True)
    image2 = np.load('MR_Images.npy', allow_pickle=True)
    BestSol = np.load('BestSol_CLS.npy', allow_pickle=True)
    Registered_Images = []
    Segmented_Images = []
    for j in range(len(image1)):
        print(j, len(image1))
        img1 = image1[j]
        img2 = image2[j]
        segmented, Registered = Model_A_3D_TRSNet(img1, img2, BestSol[4, :])
        Registered_Images.append(Registered)
        Segmented_Images.append(segmented)
    np.save('Image_Registered.npy', Registered_Images)
    np.save('Image_segmentation.npy', Segmented_Images)

plot_conv()
plot_results_Seg()
plot_Images_vs_terms()
Image_Results()
Sample_images()
