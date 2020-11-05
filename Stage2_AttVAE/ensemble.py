"""
@author: Chenggang
@github: https://github.com/MissShihongHowRU
@time: 2020-09-09 22:04
"""

import sys
sys.path.append(".")
import numpy as np
import argparse
import os
from tqdm import tqdm
import nibabel as nib
from config import config
from utils import dim_recovery, combine_labels_predicting, read_stik_to_nparray, stage2net_preprocessor
import json
from dataset import test_time_crop

def init_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', type=int, help='0 stands for majority_voting; 1 stands for averaging')
    parser.add_argument('--test', type=bool, help='make prediction on testing data', default=False)

    return parser.parse_args()


def majority_voting_for_one_patient(patient_name, models):
    """

    :param array: A slice of preds+props array along the axis 0.   Shape of input_array = (num_models + num_models)
    :return: A number indicating the result label
    """

    preds_cur_patient = []
    for model in models:
        result_path = os.path.join("../pred_stage2", model, patient_name)
        pred_cur_patient = read_stik_to_nparray(result_path)
        preds_cur_patient.append(pred_cur_patient)
    preds_cur_patient = np.array(preds_cur_patient, dtype=np.int8)
    # crop to save computational cost:
    preds_cur_patient = test_time_crop(preds_cur_patient)    # (15, 144, 192, 160)
    # determine voxels with discrimination to reduce computation
    mask_same = (preds_cur_patient[0] == preds_cur_patient[1])
    print("same_sum:{:d}".format(mask_same.sum()))
    for i in range(2, preds_cur_patient.shape[0]):
        mask_same = mask_same * (preds_cur_patient[0] == preds_cur_patient[i])
    print("same_sum:{:d}".format(mask_same.sum()))
    mask_diff = np.ones(mask_same.shape) - mask_same
    print("diff_sum:{}".format(mask_diff.sum()))
    candidates = np.array(np.where(mask_diff))
    voted_preds = preds_cur_patient[0]   # (144, 192, 160)
    # iterate (i, j, k)
    for idx in range(candidates.shape[-1]):
        coord = candidates[:, idx]
    # for i in range(preds_cur_patient.shape[1]):
    #     for j in range(preds_cur_patient.shape[2]):
    #         for k in range(preds_cur_patient.shape[3]):
        z, y, x = tuple(coord)
        label_dict = {0: 0, 1: 0, 2: 0, 4: 0}
        preds_array = preds_cur_patient[:, z, y, x]
        for pred_label in preds_array:
            label_dict[pred_label] += 1   # counting labels
        maxNum_vote = max(label_dict.values())
        majority_labels = []
        for pred_label, v in label_dict.items():
            if v == maxNum_vote:
                majority_labels.append(pred_label)
        if len(majority_labels) > 1:
            if 0 in majority_labels:  # [0,1,2]
                voted_preds[z, y, x] = 0
                continue
            elif 2 in majority_labels:
                voted_preds[z, y, x] = 2
            elif 1 in majority_labels:
                voted_preds[z, y, x] = 1
            else:
                voted_preds[z, y, x] = 4

        else:
            voted_preds[z, y, x] = majority_labels[0]
    return dim_recovery(voted_preds)


def majority_voting_for_one_patient_with_prob_comparison(patient_name, models):

    patient_name = patient_name.split(".")[0]
    prob1_path_190 = os.path.join("../pred", outp1_190_model + '_probabilityMap', patient_name + ".npy")
    prob1_path_254 = os.path.join("../pred", outp1_254_model + '_probabilityMap', patient_name + ".npy")
    prob1_path_289 = os.path.join("../pred", outp1_289_model + '_probabilityMap', patient_name + ".npy")
    probs_outp1_190 = np.load(prob1_path_190)
    probs_outp1_254 = np.load(prob1_path_254)
    probs_outp1_289 = np.load(prob1_path_289)
    probs_outp1_190 = dim_recovery(probs_outp1_190)
    probs_outp1_254 = dim_recovery(probs_outp1_254)
    probs_outp1_289 = dim_recovery(probs_outp1_289)
    preds_cur_patient = []
    # probs_cur_patient = np.empty([len(models), 128, 128, 128])
    probs_cur_patient = []
    model_idx = 0
    while model_idx < len(models):
        model = models[model_idx]
        if args.test:
            model += "_testing"
        result_path = os.path.join("../pred_stage2", model, patient_name + ".nii.gz")
        pred_cur_patient = read_stik_to_nparray(result_path)
        preds_cur_patient.append(pred_cur_patient)
        prob_path = os.path.join("../pred_stage2", model + '_probabilityMap', patient_name + ".npy")
        # read predicted prop
        prob_cur_patient = np.load(prob_path)
        assert prob_cur_patient.shape == (3, 128, 128, 128), "The shape of probability map must be (3, 128, 128, 128)"
        # probs_cur_patient.append(prob_cur_patient)  # [15 * (3, 128, 128, 128)]
        # get patch for this model
        patches_dict = read_patches_json(model)
        patch = patches_dict[patient_name]
        z_start = patch["z"]
        y_start = patch["y"]
        x_start = patch["x"]
        patch_size = patch["size"]
        if model_idx < len(models)/3:
            prob2_ = probs_outp1_190.copy()   # (3, 155, 240, 240)
        elif model_idx < len(models)/3 * 2:
            prob2_ = probs_outp1_254.copy()
        else:
            prob2_ = probs_outp1_289.copy()
        prob2_[:, z_start: z_start + patch_size, y_start: y_start + patch_size,
                                                 x_start: x_start + patch_size] = prob_cur_patient
        probs_cur_patient.append(prob2_)
        model_idx += 1
    probs_cur_patient = np.array(probs_cur_patient, dtype=np.float)    # (15, 3, 128, 128, 128)
    preds_cur_patient = np.array(preds_cur_patient, dtype=np.int8)
    preds_cur_patient = test_time_crop(preds_cur_patient)  # (12, 144, 192, 160)
    # determine voxels with discrimination to reduce computation
    mask_same = (preds_cur_patient[0] == preds_cur_patient[1])
    for i in range(2, preds_cur_patient.shape[0]):
        mask_same = mask_same * (preds_cur_patient[0] == preds_cur_patient[i])
    mask_diff = np.ones(mask_same.shape) - mask_same
    candidates = np.array(np.where(mask_diff))
    voted_preds = preds_cur_patient[0]
    # iterate (i, j, k)
    for idx in range(candidates.shape[-1]):
        coord = candidates[:, idx]
        z, y, x = tuple(coord)
        label_dict = {0: 0, 1: 0, 2: 0, 4: 0}
        preds_array = preds_cur_patient[:, z, y, x]
        for pred_label in preds_array:
            label_dict[pred_label] += 1
        maxNum_vote = max(label_dict.values())
        majority_labels = []
        for pred_label, v in label_dict.items():
            if v == maxNum_vote:
                majority_labels.append(pred_label)
        if 0 in majority_labels:  #[0,1,2]
            if len(majority_labels) == 1:
                voted_preds[z, y, x] = 0
                continue
            else:
                majority_labels.remove(0)
        if len(majority_labels) > 1:
            probs_array = probs_cur_patient[:, :, z, y, x]
            avg_props = []
            for label in majority_labels:
                if label == 1:       # [2 ,1 ,4]
                    label_idx = 1
                elif label == 2:
                    label_idx = 0
                else:
                    label_idx = 2
                prob_tobe_compared = probs_array[preds_array == label][:, label_idx]
                avg_props.append(prob_tobe_compared.mean())
            voted_preds[z, y, x] = majority_labels[np.argmax(avg_props)]
        else:
            voted_preds[z, y, x] = majority_labels[0]

    return dim_recovery(voted_preds)


def ensemble_majority_voting(models, with_prob=False):

    # get patients name
    if args.test:
        pred_path = os.path.join("../pred_stage2", models[0] + "_testing")
    else:
        pred_path = os.path.join("../pred_stage2", models[0])
    patients_name = os.listdir(pred_path)

    ensemble_process = tqdm(patients_name)
    for (i, patient) in enumerate(ensemble_process):

        ensemble_process.set_description("Processing Patient:%d" % (i))
        if with_prob:
            output_array = majority_voting_for_one_patient_with_prob_comparison(patient, models)
        else:
            output_array = majority_voting_for_one_patient(patient, models)
        patient_name = patient.split(".")[0]
        affine = nib.load(os.path.join(config["test_path"], patient_name, patient_name + '_t1.nii.gz')).affine
        output_array = output_array.swapaxes(-3, -1)  # convert channel first (SimpleTIK) to channel last (Nibabel)
        output_image = nib.Nifti1Image(output_array, affine)
        if not os.path.exists(config["prediction_dir"]):
            os.mkdir(config["prediction_dir"])
        output_image.to_filename(os.path.join(config["prediction_dir"], patient))

def read_patches_json(model):
    json_path = os.path.join("../pred_stage2", model + "_probabilityMap", 'patch_per_patient.json')
    with open(json_path, "r") as json_file:
        list_patches_with_name = json_file.readlines()
    patches_dict = {}
    for str_json in list_patches_with_name:
        patch_dict = json.loads(str_json)
        patches_dict[patch_dict["patient"]] = patch_dict
    return patches_dict

def averaging_for_one_patient(patient_name, models, threshold=0.5):

    file_path_prob_map = os.path.join(config["segmentation_map_path"] + "_probabilityMap", patient_name + ".npy")
    outp1_prob_map = np.load(file_path_prob_map)  # (3, 144, 192, 160)
    outp1_prob_map = dim_recovery(outp1_prob_map)  # (3, 155, 240, 240)
    probs_cur_patient_cropped = []
    for model in models:
        prob_path = os.path.join("../pred_stage2", model + "_probabilityMap", patient_name + ".npy")
        patches_dict = read_patches_json(model)
        patch = patches_dict[patient_name]  # {"z": 13, "y": 56, "x": 56, "size": 128, "patient": "BraTS20_Validation_106"}
        z_start = patch["z"]
        y_start = patch["y"]
        x_start = patch["x"]
        patch_size = patch["size"]
        # read predicted prop
        prob_cur_patient = np.load(prob_path)     # (3, 128, 128, 128)
        outp1_prob_map[:, z_start: z_start + patch_size, y_start: y_start + patch_size,
                                                         x_start: x_start + patch_size] = prob_cur_patient
        # crop to save computational cost:
        prob_cur_patient_cropped = test_time_crop(outp1_prob_map)  # (3, 144, 192, 160)
        probs_cur_patient_cropped.append(prob_cur_patient_cropped)
    probs_cur_patient = np.array(probs_cur_patient_cropped, dtype=np.float)  # (12, 3, 144, 192, 160)
    probs_cur_patient_averaged = np.mean(probs_cur_patient, 0)    # (3, 144, 192, 160)
    res_cur_patient = np.array(probs_cur_patient_averaged > threshold, dtype=float)  # (3, 144, 192, 160)
    res_cur_patient = combine_labels_predicting(res_cur_patient)   # (144, 192, 160)
    return dim_recovery(res_cur_patient)   #  (155, 240, 240)


def ensemble_averaging(models):
    # get patients name# (3, 155, 240, 240)
    pred_path = os.path.join("../pred_stage2", models[0])
    patients_name = os.listdir(pred_path)

    ensemble_process = tqdm(patients_name)
    for (i, patient) in enumerate(ensemble_process):

        ensemble_process.set_description("Processing Patient:%d" % (i))
        patient_name = patient.split(".")[0]
        output_array = averaging_for_one_patient(patient_name, models)
        # print(output_array.shape)  # (3, 155, 240, 240)
        # output_array = combine_labels(output_array)
        # output_array = combine_labels_predicting(output_array)
        affine = nib.load(os.path.join(config["test_path"], patient_name, patient_name + '_t1.nii.gz')).affine
        output_array = output_array.swapaxes(-3, -1)  # convert channel first (SimpleTIK) to channel last (Nibabel)
        output_image = nib.Nifti1Image(output_array, affine)
        if not os.path.exists(config["prediction_dir"]):
            os.mkdir(config["prediction_dir"])
        output_image.to_filename(os.path.join(config["prediction_dir"], patient))


# outp1_190_model = "nml4_lr_loss_crop_[214]_190_0.1633_0.9106_0.8493"
# outp1_254_model = "nml4_lr_loss_crop_[214]_254_0.1633_0.9120_0.8496"
# outp1_289_model = "nml4_lr_loss_crop_[214]_v2_289_0.1666_0.9112_0.8508"
outp1_190_model = "nml4_lr_loss_crop_[214]_190_0.1633_0.9106_0.8493_testing"
outp1_254_model = "nml4_lr_loss_crop_[214]_254_0.1633_0.9120_0.8496_testing"
outp1_289_model = "nml4_lr_loss_crop_[214]_v2_289_0.1666_0.9112_0.8508_testing"

if __name__ == "__main__":

    args = init_args()
    method = args.method

    # model_list = ["nml4_lr_loss_crop_275_0.1626_0.9142"]
    # model_list.append("nml4_lr_loss_crop_137_0.1665_0.9125")
    # model_list.append("nml4_lr_loss_crop_217_0.1648_0.9139")
    # model_list.append("nml4_lr_loss_115_0.1639_0.9135")
    # model_list.append("nml4_lr_loss_121_0.1703_0.9141")
    # model_list.append("nml4_lr_loss_155_0.1634_0.9155")
    # model_list.append("nml4_lr_loss_crop_276_trloss_0.1460_0.9326_0.8705")
    # model_list.append("nml4_lr_loss_crop_349_trloss_0.1358_0.9403_0.8822")
    # model_list.append("nml4_lr_loss_crop_394_trloss_0.1325_0.9422_0.8854")

    model_list = ["st2_multi_42_0.2040_0.8637_0.7822_0.8528"]
    model_list.append("st2_multi_54_0.2038_0.8634_0.7841_0.8534")
    model_list.append("st2_multi_57_0.2041_0.8639_0.7810_0.8524")
    model_list.append("st2_multi_66_0.2039_0.8628_0.7842_0.8531")
    model_list.append("st2_multi_69_0.2041_0.8636_0.7839_0.8535")
    model_list.append("st2_multi_40_0.2038_0.8634_0.7802_0.8519")
    model_list.append("st2_multi_53_0.2040_0.8637_0.7764_0.8510")
    model_list.append("st2_multi_59_0.2040_0.8639_0.7818_0.8524")
    model_list.append("st2_nml4_att_patch128_177_0.2015_0.8733_0.7755_0.8536")
    model_list.append("st2_nml4_att_patch128_181_0.2010_0.8719_0.7749_0.8531")
    model_list.append("st2_nml4_att_patch128_183_0.2014_0.8732_0.7744_0.8535")
    model_list.append("st2_nml4_att_patch128_210_0.2021_0.8739_0.7724_0.8529")


    model_list_190_254_289 = ['st2_multi_57_0.2041_0.8639_0.7810_0.8524_190ep']
    model_list_190_254_289.append("st2_multi_66_0.2039_0.8628_0.7842_0.8531_190ep")
    model_list_190_254_289.append("st2_multi_69_0.2041_0.8636_0.7839_0.8535_190ep")
    model_list_190_254_289.append("st2_multi_v2_67_0.2037_0.8642_0.7831_0.8531_190ep")
    model_list_190_254_289.append("st2_multi_v2_77_0.2040_0.8652_0.7798_0.8525_190ep")
    model_list_190_254_289.append("st2_multi_100_0.2053_0.8631_0.7803_0.8520_190ep")
    # model_list_190_254_289.append("st2_multi_104_0.2042_0.8635_0.7821_0.8530_190ep")
    # model_list_190_254_289.append("st2_multi_112_0.2050_0.8635_0.7843_0.8534_190ep")
    model_list_190_254_289.append("st2_multi_115_0.2053_0.8630_0.7840_0.8535_190ep")

    model_list_190_254_289.append("st2_multi_57_0.2041_0.8639_0.7810_0.8524_254ep")
    model_list_190_254_289.append("st2_multi_66_0.2039_0.8628_0.7842_0.8531_254ep")
    model_list_190_254_289.append("st2_multi_69_0.2041_0.8636_0.7839_0.8535_254ep")
    model_list_190_254_289.append("st2_multi_v2_67_0.2037_0.8642_0.7831_0.8531_254ep")
    model_list_190_254_289.append("st2_multi_v2_77_0.2040_0.8652_0.7798_0.8525_254ep")
    model_list_190_254_289.append("st2_multi_100_0.2053_0.8631_0.7803_0.8520_254ep")
    # model_list_190_254_289.append("st2_multi_104_0.2042_0.8635_0.7821_0.8530_254ep")
    # model_list_190_254_289.append("st2_multi_112_0.2050_0.8635_0.7843_0.8534_254ep")
    model_list_190_254_289.append("st2_multi_115_0.2053_0.8630_0.7840_0.8535_254ep")

    model_list_190_254_289.append("st2_multi_57_0.2041_0.8639_0.7810_0.8524_289ep")
    model_list_190_254_289.append("st2_multi_66_0.2039_0.8628_0.7842_0.8531_289ep")
    model_list_190_254_289.append("st2_multi_69_0.2041_0.8636_0.7839_0.8535_289ep")
    model_list_190_254_289.append("st2_multi_v2_67_0.2037_0.8642_0.7831_0.8531_289ep")
    model_list_190_254_289.append("st2_multi_v2_77_0.2040_0.8652_0.7798_0.8525_289ep")
    model_list_190_254_289.append("st2_multi_100_0.2053_0.8631_0.7803_0.8520_289ep")
    # model_list_190_254_289.append("st2_multi_104_0.2042_0.8635_0.7821_0.8530_289ep")
    # model_list_190_254_289.append("st2_multi_112_0.2050_0.8635_0.7843_0.8534_289ep")
    model_list_190_254_289.append("st2_multi_115_0.2053_0.8630_0.7840_0.8535_289ep")

    args = init_args()
    config["test_path"] = os.path.join(config["base_path"], "MICCAI_BraTS2020_ValidationData")
    if args.test:
        config["test_path"] = os.path.join(config["base_path"], "MICCAI_BraTS2020_TestingData")

    segmentation_map_path = "nml4_lr_loss_crop_[214]_v2_289_0.1666_0.9112_0.8508"
    config["segmentation_map_path"] = os.path.join(config["base_path"], "pred", segmentation_map_path)
    preprocessor = stage2net_preprocessor(config, patch_size=128)

    if method == 0:
        config["prediction_dir"] = os.path.join("../", "pred_stage2", "ensemble_{}_voting".format(len(model_list_190_254_289)))
        ensemble_majority_voting(model_list_190_254_289)

    elif method == 2:
        config["prediction_dir"] = os.path.join("../", "pred_stage2", "ensemble_{}_voting_withProbs".format(len(model_list_190_254_289)))
        ensemble_majority_voting(model_list_190_254_289, with_prob=True)

    elif method == 1:
        config["prediction_dir"] = os.path.join("../", "pred_stage2", "ensemble_{}_averaging".format(len(model_list_190_254_289)))
        ensemble_averaging(model_list_190_254_289)
