"""
@author: Chenggang
@github: https://github.com/MissShihongHowRU
@time: 2020-09-09 22:04
"""
import os

os.environ['PYTHONHASHSEED'] = '0'

import sys
sys.path.append(".")

import numpy as np
import random
import torch
import os
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from utils import Logger, load_old_model, poly_lr_scheduler
from train import train_epoch
from validation import val_epoch
from metrics import CombinedLoss, SoftDiceLoss
from dataset import BratsDataset
import argparse
from config import config

# set seed
seed_num = 64
np.random.seed(seed_num)
random.seed(seed_num)

def init_args():

    parser = argparse.ArgumentParser()
    # parser.add_argument('--batch_size', type=int, help='The batch size of the test images', default=2)
    parser.add_argument('-e', '--epoch', type=str, help='The number of epochs of training', default=100)
    parser.add_argument('-g', '--num_gpu', type=int, help='Can be 0, 1, 2, 4', default=2)
    parser.add_argument('-a', '--attention', type=int, help='whether to use attention blocks, 0, 1, 2', default=0)
    parser.add_argument('--concat', type=bool, help='whether to use add & concatnate connection', default=False)
    parser.add_argument('-c', '--combine', type=bool, help='whether to use newSoftDiceLoss 1+2+4', default=False)
    parser.add_argument('-d', '--dropout', type=bool, help='whether to add one dropout layer within each DownSampling operation', default=False)
    parser.add_argument('-f', '--flooding', type=bool, help='whether to apply flooding strategy during training', default=False)
    parser.add_argument('--seglabel', type=int, help='whether to train the model with 1 or all 3 labels', default=0)
    parser.add_argument('-t', '--act', type=int, help='activation function, choose between 0 and 1; 0-ReLU; 1-Sin', default=0)
    parser.add_argument('-s', '--save_folder', type=str, help='The folder for model saving', default='saved_pth')
    parser.add_argument('-p', '--pth', type=str, help='name of the saved pth file', default='')
    parser.add_argument('-i', '--image_shape', type=int,  nargs='+', help='The shape of input tensor;'
                                                                    'have to be dividable by 16 (H, W, D)',
                        default=[128, 192, 160])

    return parser.parse_args()

args = init_args()
num_epoch = args.epoch
num_gpu = args.num_gpu
batch_size = args.num_gpu
# batch_size = 1
new_SoftDiceLoss = args.combine
dropout = args.dropout
flooding = args.flooding
seglabel_idx = args.seglabel
label_list = [None, "WT", "TC", "ET"]   # None represents using all 3 labels
dice_list = [None, "dice_wt", "dice_tc", "dice_et"]
seg_label = label_list[seglabel_idx]  # used for data generation
seg_dice = dice_list[seglabel_idx]  # used for dice calculation
activation_list = ["relu", "sin"]
activation = activation_list[args.act]
save_folder = args.save_folder
pth_name = args.pth
image_shape = tuple(args.image_shape)

concat = args.concat
attention_idx = args.attention

attention = True
if attention_idx == 0:
    if concat:
        from nvnet_cat import NvNet
    elif dropout:
        from nvnet_dropout import NvNet
    else:
        from nvnet import NvNet
    attention = False
elif attention_idx == 1:
    from attVnet import AttentionVNet
elif attention_idx == 2:
    if dropout:
        from attVnet_v2_dropout import AttentionVNet
    else:
        from attVnet_v2 import AttentionVNet

config["cuda_devices"] = True
if num_gpu == 0:
    config["cuda_devices"] = None
elif num_gpu == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
elif num_gpu == 2:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
elif num_gpu == 4:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

config["batch_size"] = batch_size
config["validation_batch_size"] = batch_size
config["model_name"] = "UNetVAE-bs{}".format(config["batch_size"])  # logger
config["image_shape"] = image_shape
config["activation"] = activation
config["input_shape"] = tuple([config["batch_size"]] + [config["nb_channels"]] + list(config["image_shape"]))
config["result_path"] = os.path.join(config["base_path"], "models", save_folder)   # save models and status files
# config["saved_model_file"] = '/Users/missshihonghowru/Desktop/nyu master/brats-challenge/pth-models_status/model.pth'
config["saved_model_file"] = config["result_path"] + pth_name
config["overwrite"] = True
if pth_name:
    config["overwrite"] = False
config["epochs"] = int(num_epoch)
config["attention"] = attention
config["seg_label"] = seg_label                             # used for data generation
config["num_labels"] = 1 if config["seg_label"] else 3      # used for model constructing
config["seg_dice"] = seg_dice                               # used for dice calculation
config["new_SoftDiceLoss"] = new_SoftDiceLoss
config["flooding"] = flooding
if config["flooding"]:
    config["flooding_level"] = 0.15


def main():
    # init or load model
    print("init model with input shape", config["input_shape"])
    if config["attention"]:
        model = AttentionVNet(config=config)
    else:
        model = NvNet(config=config)
    parameters = model.parameters()
    optimizer = optim.Adam(parameters, 
                           lr=config["initial_learning_rate"],
                           weight_decay=config["L2_norm"])
    start_epoch = 1
    if config["VAE_enable"]:
        loss_function = CombinedLoss(new_loss=config["new_SoftDiceLoss"],
                                     k1=config["loss_k1_weight"], k2=config["loss_k2_weight"],
                                     alpha=config["focal_alpha"], gamma=config["focal_gamma"],
                                     focal_enable=config["focal_enable"])
    else:
        loss_function = SoftDiceLoss(new_loss=config["new_SoftDiceLoss"])

    with open('valid_list_v2.txt', 'r') as f:
        val_list = f.read().splitlines()
    # with open('train_list.txt', 'r') as f:
    with open('train_list_v2.txt', 'r') as f:
        tr_list = f.read().splitlines()

    config["training_patients"] = tr_list
    config["validation_patients"] = val_list
    # data_generator
    print("data generating")
    training_data = BratsDataset(phase="train", config=config)
    # x = training_data[0] # for test
    valildation_data = BratsDataset(phase="validate", config=config)
    train_logger = Logger(model_name=config["model_name"] + '.h5',
                          header=['epoch', 'loss', 'wt-dice', 'tc-dice', 'et-dice', 'lr'])

    if not config["overwrite"] and config["saved_model_file"] is not None:
        if not os.path.exists(config["saved_model_file"]):
            raise Exception("Invalid model path!")
        model, start_epoch, optimizer_resume = load_old_model(model, optimizer, saved_model_path=config["saved_model_file"])
        parameters = model.parameters()
        optimizer = optim.Adam(parameters,
                               lr=optimizer_resume.param_groups[0]["lr"],
                               weight_decay=optimizer_resume.param_groups[0]["weight_decay"])

    if config["cuda_devices"] is not None:
        model = model.cuda()
        loss_function = loss_function.cuda()
        model = nn.DataParallel(model)    # multi-gpu training
        for state in optimizer.state.values():
	        for k, v in state.items():
	            if isinstance(v, torch.Tensor):
	                state[k] = v.cuda()

    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=config["lr_decay"], patience=config["patience"])
    scheduler = lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=poly_lr_scheduler)  # can't restore lr correctly

    max_val_WT_dice = 0.
    max_val_AVG_dice = 0.
    for i in range(start_epoch, config["epochs"]):
        train_epoch(epoch=i, 
                    data_set=training_data, 
                    model=model,
                    criterion=loss_function, 
                    optimizer=optimizer, 
                    opt=config, 
                    logger=train_logger) 
        
        val_loss, WT_dice, TC_dice, ET_dice = val_epoch(epoch=i,
                    data_set=valildation_data,
                    model=model,
                    criterion=loss_function,
                    opt=config,
                    optimizer=optimizer,
                    logger=train_logger)

        scheduler.step()
        # scheduler.step(val_loss)
        dices = np.array([WT_dice, TC_dice, ET_dice])
        AVG_dice = dices.mean()
        if config["checkpoint"] and (WT_dice > max_val_WT_dice or AVG_dice > max_val_AVG_dice or WT_dice >= 0.912):
            max_val_WT_dice = WT_dice
            max_val_AVG_dice = AVG_dice
            # save_dir = os.path.join(config["result_path"], config["model_file"].split("/")[-1].split(".h5")[0])
            save_dir = config["result_path"]
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_states_path = os.path.join(save_dir, 'epoch_{0}_val_loss_{1:.4f}_WTdice_{2:.4f}_AVGDice:{3:.4f}.pth'.format(i, val_loss, WT_dice, AVG_dice))
            if config["cuda_devices"] is not None:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            states = {
                'epoch': i,
                'state_dict': state_dict,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_states_path)
            save_model_path = os.path.join(save_dir, "best_model.pth")
            if os.path.exists(save_model_path):
                os.system("rm "+save_model_path)
            torch.save(model, save_model_path)
        print("batch {0:d} finished, validation loss:{1:.4f}; WTDice:{2:.4f}; AVGDice:{3:.4f}".format(i, val_loss, WT_dice, AVG_dice))

if __name__ == '__main__':
    main()
