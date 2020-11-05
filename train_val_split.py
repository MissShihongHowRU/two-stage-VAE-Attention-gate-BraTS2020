import os
from Stage1_VAE.config import config
from Stage1_VAE.dataset import validation_sampling

# train-val-split 2020
if config["data_path"].split('/')[-1] == "MICCAI_BraTS2020_TrainingData":
    from pandas import read_csv
    mapping_file_path = os.path.join(config["data_path"], "name_mapping.csv")
    name_mapping = read_csv(mapping_file_path)
    HGG_patient_names = name_mapping.loc[name_mapping.Grade == "HGG", "BraTS_2020_subject_ID"].tolist()
    LGG_patient_names = name_mapping.loc[name_mapping.Grade == "LGG", "BraTS_2020_subject_ID"].tolist()

# 2018
if config["data_path"].split('/')[-1] == "MICCAI_BraTS_2018_Data_Training":
    HGG_patient_names = os.listdir(config["data_path"] + '/HGG')
    LGG_patient_names = os.listdir(config["data_path"] + '/LGG')

tr_HGG, val_HGG = validation_sampling(HGG_patient_names)
tr_LGG, val_LGG = validation_sampling(LGG_patient_names)

with open('train_list.txt', 'w') as f:
    for item in tr_HGG + tr_LGG:
        f.write("%s\n" % item)
with open('valid_list.txt', 'w') as f:
    for item in val_HGG + val_LGG:
        f.write("%s\n" % item)