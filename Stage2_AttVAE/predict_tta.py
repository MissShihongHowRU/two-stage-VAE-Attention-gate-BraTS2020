"""
@author: Chenggang
@github: https://github.com/MissShihongHowRU
@time: 2020-09-09 22:04
"""
import os
import sys
sys.path.append(".")

import numpy as np
import nibabel as nib
import argparse
import torch
from tqdm import tqdm
from dataset import PatchDataset
from attVnet_add import AttentionVNet
from config import config
from pandas import read_csv
from utils import combine_labels_predicting, read_stik_to_nparray, stage2net_preprocessor
import json

def init_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--num_gpu', type=int, help='Can be 0, 1, 2, 4', default=2)
    parser.add_argument('-s', '--save_folder', type=str, help='The folder of the saved model', default='saved_pth')
    parser.add_argument('-f', '--checkpoint_file', type=str, help='name of the saved pth file', default='')
    parser.add_argument('-m', '--map_path', type=str, help='The folder including segmentation map for predicting', default='nml4_lr_loss_crop_[214]_190_0.1633_0.9106_0.8493')
    parser.add_argument('--test', type=bool, help='make prediction on testing data', default=False)
    parser.add_argument('--seglabel', type=int, help='whether to train the model with 1 or all 3 labels', default=0)
    parser.add_argument('-t', '--tta', type=bool, help='Whether to implement test-time augmentation;', default=False)

    return parser.parse_args()


args = init_args()
num_gpu = args.num_gpu
tta = args.tta
segmentation_map_path = args.map_path
config["predict_from_test_data"] = args.test
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

patch_size = 128
image_shape = tuple([patch_size] * 3)
if config["predict_from_test_data"]:
    segmentation_map_path += "_testing"

config["image_shape"] = image_shape
config["checkpoint_file"] = args.checkpoint_file
config["segmentation_map_path"] = os.path.join(config["base_path"], "pred", segmentation_map_path)
config["checkpoint_path"] = os.path.join(config["base_path"], "model", args.save_folder)
config['saved_model_path'] = os.path.join(config["checkpoint_path"], config["checkpoint_file"])
# config["prediction_dir"] = os.path.abspath("./prediction/")
config["prediction_dir"] = os.path.join(config["base_path"], "pred_stage2", config["checkpoint_file"].split(".pth")[0])
config["load_from_data_parallel"] = True  # Load model trained on multi-gpu to predict on single gpu.
config["test_path"] = os.path.join(config["base_path"], "data", "MICCAI_BraTS2020_ValidationData")
if config["predict_from_test_data"]:
    config["test_path"] = os.path.join(config["base_path"], "data", "MICCAI_BraTS2020_TestingData")
config["input_shape"] = tuple([config["batch_size"]] + [config["nb_channels"]] + list(config["image_shape"]))
config["VAE_enable"] = False
config["seg_label"] = seg_label                             # used for data generation
config["num_labels"] = 1 if config["seg_label"] else 3      # used for model constructing
config["seg_dice"] = seg_dice
config["predict_from_train_data"] = False


config["activation"] = "relu"
if "sin" in config["checkpoint_file"]:
    config["activation"] = "sin"

config["concat"] = False
if "cat" in config["checkpoint_file"]:
    config["concat"] = True

config["attention"] = False
if "att" in config["checkpoint_file"]:
    config["attention"] = True


def init_model_from_states(config):

    print("Init model...")
    model = AttentionVNet(config=config)

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

    # config["prediction_dir"] += "_190ep"   # indicate which outp1 the prediction is based on.
    # config["prediction_dir"] += "_254ep"   # indicate which outp1 the prediction is based on.
    config["prediction_dir"] += "_289ep"   # indicate which outp1 the prediction is based on.
    if config["predict_from_test_data"]:
        config["prediction_dir"] += "_testing"
    if not os.path.exists(config["prediction_dir"]):
        os.mkdir(config["prediction_dir"])

    preprocessor = stage2net_preprocessor(config, patch_size=128, overlap=112)
    data_set = PatchDataset(phase="test", config=config, preprocessor=preprocessor)
    valildation_loader = torch.utils.data.DataLoader(dataset=data_set,
                                                     batch_size=config["batch_size"],
                                                     shuffle=False,
                                                     pin_memory=True)
    predict_process = tqdm(valildation_loader)
    for idx, (inputs, patch) in enumerate(predict_process):

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
        probsMap_array = output_array[:, :3, :, :, :]  # (2, 7, 128, 128, 128)
        preds_array = np.array(probsMap_array > 0.5, dtype=float)  # (2, 3, 128, 128, 128)
        preds_array = combine_labels_predicting(preds_array) # (2, 128, 128, 128)

        for i in range(config["batch_size"]):
            file_idx = idx * config["batch_size"] + i
            if file_idx < len(name_list):

                z_start = patch["z"][i]
                y_start = patch["y"][i]
                x_start = patch["x"][i]
                patch_size = patch["size"][i]
                patch_i = {k: int(v.cpu().numpy()[i]) for k, v in patch.items()}

                patient_filename = name_list[file_idx]

                outp1_path = os.path.join(config["segmentation_map_path"], patient_filename + '.nii.gz')

                outp1_npy = read_stik_to_nparray(outp1_path)   # (1, 155, 240, 240)
                outp1_npy[z_start: z_start+patch_size, y_start: y_start+patch_size,
                                                        x_start: x_start+patch_size] = preds_array[i]

                outp1_npy = outp1_npy.swapaxes(-3, -1)  # convert channel first (SimpleTIK) to channel last (Nibabel)

                affine = nib.load(os.path.join(config["test_path"], patient_filename, patient_filename + '_t1.nii.gz')).affine
                output_image = nib.Nifti1Image(outp1_npy, affine)
                if not os.path.exists(config["prediction_dir"]):
                    os.mkdir(config["prediction_dir"])
                output_image.to_filename(os.path.join(config["prediction_dir"], patient_filename + '.nii.gz'))
                propbsMap_dir = config["prediction_dir"] + "_probabilityMap"
                if not os.path.exists(propbsMap_dir):
                    os.mkdir(propbsMap_dir)
                np.save(os.path.join(propbsMap_dir, patient_filename + ".npy"), probsMap_array[i])
                patch_i["patient"] = patient_filename
                str_js = json.dumps(patch_i) + "\n"    # TypeError: Object of type Tensor is not JSON serializable
                propbsMap_json_dir = os.path.join(propbsMap_dir, 'patch_per_patient.json')
                with open(propbsMap_json_dir, 'a') as js_f:
                    js_f.write(str_js)

if __name__ == "__main__":

    model = init_model_from_states(config)
    if config["predict_from_test_data"]:
        mapping_file_path = os.path.join(config["test_path"], "survival_evaluation.csv")
        name_mapping = read_csv(mapping_file_path)
        val_list = name_mapping["BraTS20ID"].tolist()
    else:
        mapping_file_path = os.path.join(config["test_path"], "name_mapping_validation_data.csv")
        name_mapping = read_csv(mapping_file_path)
        val_list = name_mapping["BraTS_2020_subject_ID"].tolist()

    predict(val_list, model)
