"""
@author: Chenggang
@github: https://github.com/MissShihongHowRU
@time: 2020-09-09 22:04
"""
import sys
sys.path.append(".")

import os
import numpy as np
import nibabel as nib
import argparse
import torch
from tqdm import tqdm
from dataset import BratsDataset
from config import config
from pandas import read_csv
from utils import combine_labels_predicting, dim_recovery

def init_args():

    parser = argparse.ArgumentParser()
    # parser.add_argument('-a', '--attention', type=int, help='choose from 0, 1, 2', default=0)
    parser.add_argument('-g', '--num_gpu', type=int, help='Can be 0, 1, 2, 4', default=2)
    parser.add_argument('-s', '--save_folder', type=str, help='The folder of the saved model', default='saved_pth')
    parser.add_argument('-f', '--checkpoint_file', type=str, help='name of the saved pth file', default='')
    parser.add_argument('--train', type=bool, help='make prediction on training data', default=False)
    parser.add_argument('--test', type=bool, help='make prediction on testing data', default=False)
    parser.add_argument('--seglabel', type=int, help='whether to train the model with 1 or all 3 labels', default=0)
    parser.add_argument('-i', '--image_shape', type=int,  nargs='+', help='The shape of input tensor;'
                                                                    'have to be dividable by 16 (H, W, D)',
                        default=[128, 192, 160])
    parser.add_argument('-t', '--tta', type=bool, help='Whether to implement test-time augmentation;', default=False)

    return parser.parse_args()


args = init_args()
num_gpu = args.num_gpu
tta = args.tta
config["cuda_devices"] = True
if num_gpu == 0:
    config["cuda_devices"] = None
elif num_gpu == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
elif num_gpu == 2:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
elif num_gpu == 4:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
config["batch_size"] = args.num_gpu
seglabel_idx = args.seglabel
label_list = [None, "WT", "TC", "ET"]   # None represents using all 3 labels
dice_list = [None, "dice_wt", "dice_tc", "dice_et"]
seg_label = label_list[seglabel_idx]  # used for data generation
seg_dice = dice_list[seglabel_idx]  # used for dice calculation
config["image_shape"] = args.image_shape
config["checkpoint_file"] = args.checkpoint_file
config["checkpoint_path"] = os.path.join(config["base_path"], "models", args.save_folder)
config['saved_model_path'] = os.path.join(config["checkpoint_path"], config["checkpoint_file"])
config["prediction_dir"] = os.path.join(config["base_path"], "pred", config["checkpoint_file"].split(".pth")[0])
config["load_from_data_parallel"] = True  # Load model trained on multi-gpu to predict on single gpu.
config["predict_from_train_data"] = args.train
config["predict_from_test_data"] = args.test
config["test_path"] = os.path.join(config["base_path"], "data", "MICCAI_BraTS2020_ValidationData")
if config["predict_from_test_data"]:
    config["test_path"] = os.path.join(config["base_path"], "data", "MICCAI_BraTS2020_TestingData")
if config["predict_from_train_data"]:
    config["test_path"] = os.path.join(config["base_path"], "data", "MICCAI_BraTS2020_TrainingData")
config["input_shape"] = tuple([config["batch_size"]] + [config["nb_channels"]] + list(config["image_shape"]))
config["VAE_enable"] = False
config["seg_label"] = seg_label                             # used for data generation
config["num_labels"] = 1 if config["seg_label"] else 3      # used for model constructing
config["seg_dice"] = seg_dice                               # used for dice calculation

config["activation"] = "relu"
if "sin" in config["checkpoint_file"]:
    config["activation"] = "sin"

config["concat"] = False
if "cat" in config["checkpoint_file"]:
    config["concat"] = True

config["attention"] = False
if "att" in config["checkpoint_file"]:
    config["attention"] = True

from nvnet import NvNet

def init_model_from_states(config):

    print("Init model...")
    model = NvNet(config=config)
    if config["cuda_devices"] is not None:
        if num_gpu > 0:
            model = torch.nn.DataParallel(model)   # multi-gpu inference
        model = model.cuda()
    checkpoint = torch.load(config['saved_model_path'], map_location='cpu')
    state_dict = checkpoint["state_dict"]
    if not config["load_from_data_parallel"]:
        model.load_state_dict(state_dict)
    else:
        from collections import OrderedDict     # Load state_dict from checkpoint model trained by multi-gpu
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if not "vae" in k:    # disable the vae path
                if "module." in k:
                    new_state_dict[k] = v
                # name = k[7:]
                else:
                    name = "module." + k    # fix the bug of missing keywords caused by data parallel
                    new_state_dict[name] = v

        model.load_state_dict(new_state_dict)

    return model


def test_time_flip_recovery(imgs_array, tta_idx):
    if tta_idx == 0:  # [0, 0, 0]
        return imgs_array
    if tta_idx == 1:  # [1, 0, 0]
        return imgs_array[:, :, ::-1, :, :]
    if tta_idx == 2:  # [0, 1, 0]
        return imgs_array[:, :, :, ::-1, :]
    if tta_idx == 3:  # [0, 0, 1]
        return imgs_array[:, :, :, :, ::-1]
    if tta_idx == 4:  # [1, 1, 0]
        return imgs_array[:, :, ::-1, ::-1, :]
    if tta_idx == 5:  # [1, 0, 1]
        return imgs_array[:, :, ::-1, :, ::-1]
    if tta_idx == 6:  # [0, 1, 1]
        return imgs_array[:, :, :, ::-1, ::-1]
    if tta_idx == 7:  # [1, 1, 1]
        return imgs_array[:, :, ::-1, ::-1, ::-1]


def predict(name_list, model):

    model.eval()
    config["test_patients"] = name_list
    # config["tta_idx"] = 0   # 0 indices no test-time augmentation;
    if not os.path.exists(config["prediction_dir"]):
        os.mkdir(config["prediction_dir"])

    tmp_dir = "../tmp_result_{}".format(config["checkpoint_file"][:-4])
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    # For testing time data augment
    tta_idx_limit = 8 if tta else 1
    for tta_idx in range(tta_idx_limit):
        config["tta_idx"] = tta_idx
        if tta:
            print("starting evaluation of the {} mirror flip of Test-Time-Augmentation".format(tta_idx))
        data_set = BratsDataset(phase="test", config=config)
        valildation_loader = torch.utils.data.DataLoader(dataset=data_set,
                                                         batch_size=config["batch_size"],
                                                         shuffle=False,
                                                         pin_memory=True)
        predict_process = tqdm(valildation_loader)
        for idx, inputs in enumerate(predict_process):
            if idx > 0:
                predict_process.set_description("processing {} picture".format(idx))

            if config["cuda_devices"] is not None:
                inputs = inputs.type(torch.FloatTensor)
                inputs = inputs.cuda()
            with torch.no_grad():
                if config["VAE_enable"]:
                    outputs, distr = model(inputs)
                else:
                    outputs = model(inputs)

            output_array = np.array(outputs.cpu())  # can't convert tensor in GPU directly
            output_array = output_array[:, :3, :, :, :]  # (2, 7, 128, 192, 160)
            print(output_array.shape)
            output_array = test_time_flip_recovery(output_array, config["tta_idx"])
            # save to tmp
            for i in range(config["batch_size"]):
                file_idx = idx * config["batch_size"] + i
                if file_idx < len(name_list):
                    patient_filename = name_list[file_idx]
                    np.save(os.path.join(tmp_dir, "flip_{}_{}.npy".format(config["tta_idx"], patient_filename)), output_array[i])
    # after all flips
    if tta:
        config["prediction_dir"] += "_TTA"
    if config["predict_from_train_data"]:
        config["prediction_dir"] += "_train"
    if config["predict_from_test_data"]:
        config["prediction_dir"] += "_testing"
    if not os.path.exists(config["prediction_dir"]):
        os.mkdir(config["prediction_dir"])
    for patient_filename in name_list:
        flip_arrays = []
        for tta_idx in range(tta_idx_limit):
            flip_array = np.load(os.path.join(tmp_dir, "flip_{}_{}.npy".format(config["tta_idx"], patient_filename)))
            flip_arrays.append(flip_array)
        probsMap_array = np.array(flip_arrays).mean(axis=0)
        preds_array = np.array(probsMap_array > 0.5, dtype=float)  # (1, 3, 128, 192, 160)
        preds_array = dim_recovery(preds_array)   # (1, 3, 155, 240, 240)
        preds_array = preds_array.swapaxes(-3, -1)  # convert channel first (SimpleTIK) to channel last (Nibabel)
        preds_array = combine_labels_predicting(preds_array)

        affine = nib.load(os.path.join(config["test_path"], patient_filename, patient_filename + '_t1.nii.gz')).affine
        output_image = nib.Nifti1Image(preds_array, affine)
        output_image.to_filename(os.path.join(config["prediction_dir"], patient_filename + '.nii.gz'))
        propbsMap_dir = config["prediction_dir"] + "_probabilityMap"
        if not os.path.exists(propbsMap_dir):
            os.mkdir(propbsMap_dir)
        np.save(os.path.join(propbsMap_dir, patient_filename + ".npy"), probsMap_array)

    os.system("rm -r " + tmp_dir)

    
if __name__ == "__main__":

    model = init_model_from_states(config)

    if config["predict_from_test_data"]:
        mapping_file_path = os.path.join(config["test_path"], "survival_evaluation.csv")
        name_mapping = read_csv(mapping_file_path)
        val_list = name_mapping["BraTS20ID"].tolist()
    else:
        if config["predict_from_train_data"]:
            mapping_file_path = os.path.join(config["test_path"], "name_mapping.csv")
        else:
            mapping_file_path = os.path.join(config["test_path"], "name_mapping_validation_data.csv")
        name_mapping = read_csv(mapping_file_path)
        val_list = name_mapping["BraTS_2020_subject_ID"].tolist()

    predict(val_list, model)
