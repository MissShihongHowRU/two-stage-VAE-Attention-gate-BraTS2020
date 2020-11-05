"""
@author: Chenggang
@github: https://github.com/MissShihongHowRU
@time: 2020-09-09 22:04
"""
import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np
import os
import SimpleITK as sitk
from dataset import BratsDataset
from config import config
from tqdm import tqdm
from dataset import preprocess_label
from utils import calculate_accuracy

def evaluate(data_set, folder="nml2_lr_cat2_loss_179_0.1657_0.9138train"):

    folder_path = os.path.join("../pred", folder)
    pred_outputs = os.listdir(folder_path)
    excel_name = os.path.join("../pred", folder + '.xlsx')
    df = pd.DataFrame(columns=["ET", "WT", "TC"], index=1 + np.arange(len(pred_outputs)))
    evaluation_loader = torch.utils.data.DataLoader(dataset=data_set,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    pin_memory=True)

    eval_process = tqdm(evaluation_loader)
    for i, (_, targets) in enumerate(eval_process):

        eval_process.set_description("Processing Patient:%d" % (i))
        # read preds
        pred_name = config["validation_patients"][i] + ".nii.gz"
        cur_pred_output = os.path.join(folder_path, pred_name)
        sitkImage = sitk.ReadImage(cur_pred_output)
        output_with_oriLabel = sitk.GetArrayFromImage(sitkImage)
        output = preprocess_label(output_with_oriLabel)
        acc, _ = calculate_accuracy(torch.Tensor(output[np.newaxis,:,:,:,:]), targets)
        df.loc[i, "WT"] = acc["dice_wt"].item()
        df.loc[i, "TC"] = acc["dice_tc"].item()
        df.loc[i, "ET"] = acc["dice_et"].item()

    print(round(df["WT"].mean(), 4))
    df.to_excel(excel_name, index=None)


if __name__ == "__main__":

    config["test_path"] = os.path.join(config["base_path"], "data", "MICCAI_BraTS2020_TrainingData")
    mapping_file_path = os.path.join(config["test_path"], "name_mapping.csv")
    name_mapping = pd.read_csv(mapping_file_path)
    config["validation_patients"] = name_mapping["BraTS_2020_subject_ID"].tolist()
    # config["validation_patients"] = ["Brats18_CBICA_AXL_1"]
    config["seg_label"] = None
    config["input_shape"] = None
    pred_name = config["validation_patients"]
    evaluation_data = BratsDataset(phase="evaluation", config=config)
    evaluate(evaluation_data)