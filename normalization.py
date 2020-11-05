"""
@author: Chenggang
@github: https://github.com/MissShihongHowRU
@time: 2020-09-09 20:36
"""
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Pool
import os
import time
import pandas as pd
import argparse
from Stage2_AttVAE.config import config



def init_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--data_path', type=str, help='The data path')
    parser.add_argument('-y', '--year', type=int, help='s', default=2020)

    return parser.parse_args()


def get_list_of_files(base_dir):
    """
    returns a list of lists containing the filenames. The outer list contains all training examples. Each entry in the
    outer list is again a list pointing to the files of that training example in the following order:
    T1, T1c, T2, FLAIR, segmentation
    :param base_dir:
    :return:
    """
    list_of_lists = []
    for glioma_type in ['HGG', 'LGG']:
        current_directory = join(base_dir, glioma_type)
        patients = subfolders(current_directory, join=False)
        for p in patients:
            patient_directory = join(current_directory, p)
            t1_file = join(patient_directory, p + "_t1.nii.gz")
            t1c_file = join(patient_directory, p + "_t1ce.nii.gz")
            t2_file = join(patient_directory, p + "_t2.nii.gz")
            flair_file = join(patient_directory, p + "_flair.nii.gz")
            seg_file = join(patient_directory, p + "_seg.nii.gz")
            this_case = [t1_file, t1c_file, t2_file, flair_file, seg_file]
            assert all((isfile(i) for i in this_case)), "some file is missing for patient %s; make sure the following " \
                                                        "files are there: %s" % (p, str(this_case))

            list_of_lists.append(this_case)

    print("Found {} patients".format(len(list_of_lists)))
    return list_of_lists


def get_list_of_files_2020(current_directory, patients, mode="training"):
    """
    returns a list of lists containing the filenames. The outer list contains all training examples. Each entry in the
    outer list is again a list pointing to the files of that training example in the following order:
    T1, T1c, T2, FLAIR, segmentation
    :param base_dir:
    :return:
    """
    list_of_lists = []
    # patients = subfolders(current_directory, join=False)
    for p in patients:
        patient_directory = join(current_directory, p)
        t1_file = join(patient_directory, p + "_t1.nii.gz")
        t1c_file = join(patient_directory, p + "_t1ce.nii.gz")
        t2_file = join(patient_directory, p + "_t2.nii.gz")
        flair_file = join(patient_directory, p + "_flair.nii.gz")
        if mode == "training":
            seg_file = join(patient_directory, p + "_seg.nii.gz")
            this_case = [t1_file, t1c_file, t2_file, flair_file, seg_file]
        else:
            this_case = [t1_file, t1c_file, t2_file, flair_file]
        assert all((isfile(i) for i in this_case)), "some file is missing for patient %s; make sure the following " \
                                                    "files are there: %s" % (p, str(this_case))

        list_of_lists.append(this_case)

    print("Found {} patients".format(len(list_of_lists)))
    return list_of_lists


def load_and_preprocess(case, patient_name, output_folder):
    """
    loads, preprocesses and saves a case
    This is what happens here:
    1) load all images and stack them to a 4d array
    2) crop to nonzero region, this removes unnecessary zero-valued regions and reduces computation time
    3) normalize the nonzero region with its mean and standard deviation
    4) save 4d tensor as numpy array. Also save metadata required to create niftis again (required for export
    of predictions)

    :param case:
    :param patient_name:
    :return:
    """
    # load SimpleITK Images
    imgs_sitk = [sitk.ReadImage(i) for i in case]

    # get pixel arrays from SimpleITK images
    imgs_npy = [sitk.GetArrayFromImage(i) for i in imgs_sitk]

    # now stack the images into one 4d array, cast to float because we will get rounding problems if we don't
    imgs_npy = np.concatenate([i[None] for i in imgs_npy]).astype(np.float32)     #(5, 155, 240, 240)

    nonzero_masks = [i != 0 for i in imgs_npy[:-1]]
    brain_mask = np.zeros(imgs_npy.shape[1:], dtype=bool)
    for i in range(len(nonzero_masks)):
        brain_mask = brain_mask | nonzero_masks[i]  # 1488885;  # 1490852;  # 1492561;  #1495212

    # now normalize each modality with its mean and standard deviation (computed within the brain mask)
    for i in range(len(imgs_npy) - 1):   # 158
        mean = imgs_npy[i][brain_mask].mean()
        std = imgs_npy[i][brain_mask].std()
        imgs_npy[i] = (imgs_npy[i] - mean) / (std + 1e-8)
        imgs_npy[i][brain_mask == 0] = 0
        print(imgs_npy[i].mean())

    # now save as npy
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    np.save(join(output_folder, patient_name + ".npy"), imgs_npy)

    print(patient_name)

def load_and_preprocess_val(case, patient_name, output_folder):

    # load images using nibabel
    imgs_nib = [nib.load(i) for i in case]
    imgs_sitk = [sitk.ReadImage(i) for i in case]

    # get pixel arrays from nib object
    # imgs_npy = [i.get_fdata() for i in imgs_nib]
    imgs_npy = [sitk.GetArrayFromImage(i) for i in imgs_sitk]

    # get affine information
    affines = [i.affine for i in imgs_nib]

    # now stack the images into one 4d array, cast to float because we will get rounding problems if we don't
    imgs_npy = np.concatenate([i[None] for i in imgs_npy]).astype(np.float32)
    # (4, 155, 240, 240) for STik; (4, 240, 240, 155) for nil
    # get affine information
    affines = np.concatenate([i[None] for i in affines]).astype(np.float32)

    # now we create a brain mask that we use for normalization
    nonzero_masks = [i != 0 for i in imgs_npy]
    brain_mask = np.zeros(imgs_npy.shape[1:], dtype=bool)
    for i in range(len(nonzero_masks)):
        brain_mask = brain_mask | nonzero_masks[i]  # 1488885;  # 1490852;  # 1492561;  #1495212

    # now normalize each modality with its mean and standard deviation (computed within the brain mask)
    for i in range(len(imgs_npy)):   # 158
        mean = imgs_npy[i][brain_mask].mean()
        std = imgs_npy[i][brain_mask].std()
        imgs_npy[i] = (imgs_npy[i] - mean) / (std + 1e-8)
        imgs_npy[i][brain_mask == 0] = 0

    # now save as npy
    affine_output_folder = output_folder[:-4] + '/affine'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if not os.path.exists(affine_output_folder):
        os.mkdir(affine_output_folder)
    np.save(join(output_folder, patient_name + ".npy"), imgs_npy)
    # np.save(join(affine_output_folder, patient_name + ".npy"), affines)

    print(patient_name)

if __name__ == "__main__":

    args = init_args()
    if args.year == 2018:

        data_file_path = "data/MICCAI_BraTS_2018_Data_Training"
        npy_normalized_folder = join(data_file_path, "npy")
        list_of_lists = get_list_of_files(data_file_path)
        patient_names = [i[0].split("/")[-2] for i in list_of_lists]
        # load_and_preprocess(HGG, patient_names, data_file_path)
        p = Pool(processes=8)   #num_threads_for_brats_example
        t0 = time.time()
        print("job starts")
        p.starmap(load_and_preprocess, zip(list_of_lists, patient_names, [npy_normalized_folder] * len(list_of_lists)))
        print("finished; costs {}s".format(time.time() - t0))
        p.close()
        p.join()

    if args.year == 2020:

        # use this code block to preprocess the Brats 2020 data
        args = init_args()
        data_file_path = "data/MICCAI_BraTS2020_TrainingData"
        npy_normalized_folder = join(data_file_path, "npy")
        mapping_file_path = join(data_file_path, "name_mapping.csv")
        name_mapping = pd.read_csv(mapping_file_path)
        HGG = name_mapping.loc[name_mapping.Grade == "HGG", "BraTS_2020_subject_ID"].tolist()
        LGG = name_mapping.loc[name_mapping.Grade == "LGG", "BraTS_2020_subject_ID"].tolist()
        patients = HGG + LGG
        list_of_lists = get_list_of_files_2020(data_file_path, patients)
        # load_and_preprocess(HGG, patient_names, data_file_path)
        p = Pool(processes=8)   #num_threads_for_brats_example
        t0 = time.time()
        print("job starts")
        p.starmap(load_and_preprocess, zip(list_of_lists, patients, [npy_normalized_folder] * len(list_of_lists)))
        print("finished; costs {}s".format(time.time() - t0))
        p.close()
        p.join()

    if args.year == 202001 or args.year == 202002:

        # use this code block to preprocess the Brats 2020 data
        args = init_args()
        # data_file_path = args.data_path
        if args.year == 202001:
            data_file_path = "data/MICCAI_BraTS2020_ValidationData"
            mapping_file_path = join(data_file_path, "name_mapping_validation_data.csv")
        if args.year == 202002:
            data_file_path = "data/MICCAI_BraTS2020_TestingData"
            data_file_path = os.path.join(config["base_path"], "MICCAI_BraTS2020_TestingData")
            mapping_file_path = join(data_file_path, "survival_evaluation.csv")
        npy_normalized_folder = join(data_file_path, "npy")
        name_mapping = pd.read_csv(mapping_file_path)
        patients = name_mapping["BraTS20ID"].tolist()
        list_of_lists = get_list_of_files_2020(data_file_path, patients, mode="validation")
        # load_and_preprocess(HGG, patient_names, data_file_path)
        p = Pool(processes=8)   #num_threads_for_brats_example
        t0 = time.time()
        print("job starts")
        p.starmap(load_and_preprocess_val, zip(list_of_lists, patients, [npy_normalized_folder] * len(list_of_lists)))
        print("finished; costs {}s".format(time.time() - t0))
        p.close()
        p.join()
