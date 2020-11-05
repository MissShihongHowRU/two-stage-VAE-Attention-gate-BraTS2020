"""
@author: Chenggang
@github: https://github.com/MissShihongHowRU
@time: 2020-09-09 22:04
"""
import pickle
import torch
import tensorboardX
import numpy as np
from collections import OrderedDict
import SimpleITK as sitk
from tqdm import tqdm
import os

def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        print("val:%.4f"%val)
        self.sum += val * n
        print("sum:%.4f"%self.sum)
        self.count += n
        print("cnt:%d"%self.count)
        self.avg = self.sum / self.count
        print("avg:%.4f"%self.avg)


class Logger(object):

    def __init__(self, model_name, header):
        self.header = header
        self.writer = tensorboardX.SummaryWriter("./runs/"+model_name.split("/")[-1].split(".h5")[0])

    def __del(self):
        self.writer.close()

    def log(self, phase, values):
        epoch = values['epoch']
        
        for col in self.header[1:]:
            self.writer.add_scalar(phase+"/"+col, float(values[col]), int(epoch))


def combine_labels(labels):
    """
    Combine wt, tc, et into WT; tc, et into TC; et into ET
    :param labels: torch.Tensor of size (bs, 3, ?,?,?); ? is the crop size
    :return:
    """
    whole_tumor = labels[:, :3, :, :, :].sum(1)  # could have 2 or 3
    tumor_core = labels[:, 1:3, :, :, :].sum(1)
    enhanced_tumor = labels[:, 2:3, :, :, :].sum(1)
    whole_tumor[whole_tumor != 0] = 1
    tumor_core[tumor_core != 0] = 1
    enhanced_tumor[enhanced_tumor != 0] = 1
    return whole_tumor, tumor_core, enhanced_tumor  # (bs, ?, ?, ?)


def calculate_accuracy(outputs, targets):
    return dice_coefficient(outputs, targets)


def dice_coefficient(outputs, targets, threshold=0.5, eps=1e-8):  # 搞三个dice看 每个label; 不要做soft dice
    # batch_size = targets.size(0)
    y_pred = outputs[:, :3, :, :, :]  # targets[0,:3,:,:,:]
    y_truth = targets[:, :3, :, :, :]
    y_pred = y_pred > threshold
    y_pred = y_pred.type(torch.FloatTensor)
    wt_pred, tc_pred, et_pred = combine_labels(y_pred)
    wt_truth, tc_truth, et_truth = combine_labels(y_truth)
    res = dict()
    res["dice_wt"] = dice_coefficient_single_label(wt_pred, wt_truth, eps)
    res["dice_tc"] = dice_coefficient_single_label(tc_pred, tc_truth, eps)
    res["dice_et"] = dice_coefficient_single_label(et_pred, et_truth, eps)

    return res


def calculate_accuracy_singleLabel(outputs, targets, threshold=0.5, eps=1e-8):

    y_pred = outputs[:, 0, :, :, :]  # targets[0,:3,:,:,:]
    y_truth = targets[:, 0, :, :, :]
    y_pred = y_pred > threshold
    y_pred = y_pred.type(torch.FloatTensor)
    res = dice_coefficient_single_label(y_pred, y_truth, eps)
    return res


def dice_coefficient_single_label(y_pred, y_truth, eps):
    # batch_size = y_pred.size(0)
    intersection = torch.sum(torch.mul(y_pred, y_truth), dim=(-3, -2, -1)) + eps / 2  # axis=?, (bs, 1)
    union = torch.sum(y_pred, dim=(-3,-2,-1)) + torch.sum(y_truth, dim=(-3,-2,-1)) + eps  # (bs, 1)
    dice = 2 * intersection / union
    return dice.mean()
    # return dice / batch_size

def calculate_accuracy_clf(y_pred, y_truth):

    # y_pred = np.array(y_pred)   # RuntimeError: Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
    y_pred = y_pred.detach().numpy()
    y_truth = y_truth.detach().numpy()  # (2, )
    acc = (y_pred == y_truth).mean()
    return acc


def load_old_model(model, optimizer, saved_model_path, data_paralell=True):
    print("Constructing model from saved file... ")
    checkpoint = torch.load(saved_model_path, map_location='cpu')
    epoch = checkpoint["epoch"]
    if data_paralell:
        state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():  # remove "module."
            if "module." in k:
                node_name = k[7:]

            else:
                node_name = k
            state_dict[node_name] = v
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return model, epoch, optimizer


def combine_labels_predicting(output_array):
    """
    # (1, 3, 240, 240, 155)
    :param output_array: output of the model containing 3 seperated labels (3 channels)
    :return: res_array: conbined labels (1 channel)
    """
    shape = output_array.shape[-3:]
    if len(output_array.shape) == 5:
        bs = output_array.shape[0]
        res_array = np.zeros((bs, ) + shape)
        res_array[output_array[:, 0, :, :, :] == 1] = 2
        res_array[output_array[:, 1, :, :, :] == 1] = 1
        res_array[output_array[:, 2, :, :, :] == 1] = 4
    elif len(output_array.shape) == 4:
        res_array = np.zeros(shape)
        res_array[output_array[0, :, :, :] == 1] = 2
        res_array[output_array[1, :, :, :] == 1] = 1
        res_array[output_array[2, :, :, :] == 1] = 4
    return res_array


def dim_recovery(img_array, orig_shape=(155, 240, 240)):
    """
    used when doing inference
    :param img_array:
    :param orig_shape:
    :return:
    """
    crop_shape = np.array(img_array.shape[-3:])
    center = np.array(orig_shape) // 2
    lower_limits = center - crop_shape // 2
    upper_limits = center + crop_shape // 2
    if len(img_array.shape) == 5:
        bs, num_labels = img_array.shape[:2]
        res_array = np.zeros((bs, num_labels) + orig_shape)
        res_array[:, :, lower_limits[0]: upper_limits[0],
                        lower_limits[1]: upper_limits[1], lower_limits[2]: upper_limits[2]] = img_array
    if len(img_array.shape) == 4:
        num_labels = img_array.shape[0]
        res_array = np.zeros((num_labels, ) + orig_shape)
        res_array[:, lower_limits[0]: upper_limits[0],
                     lower_limits[1]: upper_limits[1], lower_limits[2]: upper_limits[2]] = img_array

    if len(img_array.shape) == 3:
        res_array = np.zeros(orig_shape)
        res_array[lower_limits[0]: upper_limits[0],
            lower_limits[1]: upper_limits[1], lower_limits[2]: upper_limits[2]] = img_array

    return res_array


def read_stik_to_nparray(gz_path):
    sitkImage = sitk.ReadImage(gz_path)
    nparray = sitk.GetArrayFromImage(sitkImage)
    return nparray


def poly_lr_scheduler(epoch, num_epochs=300, power=0.9):
    return (1 - epoch/num_epochs)**power


def poly_lr_scheduler_multi(epoch, num_epochs=200, power=0.9): # 1.5
    return (1 - epoch/num_epochs)**power

def poly_lr_scheduler_clf(epoch, num_epochs=100, power=0.9):
    return (1 - epoch/num_epochs)**power

def central_area_crop(imgs_array, crop_size=(144, 192, 160)):
    """
    crop the test image around the center; default crop_zise change from (128, 192, 160) to (144, 192, 160)
    :param imgs_array:
    :param crop_size:
    :return: image with the size of crop_size
    """
    orig_shape = np.array(imgs_array.shape)
    crop_shape = np.array(crop_size)
    center = orig_shape // 2
    lower_limits = center - crop_shape // 2  # (13, 24, 40) (5, 24, 40)
    upper_limits = center + crop_shape // 2  # (141, 216, 200) (149, 216, 200）
    # upper_limits = lower_limits + crop_shape
    imgs_array = imgs_array[lower_limits[0]: upper_limits[0],
                 lower_limits[1]: upper_limits[1], lower_limits[2]: upper_limits[2]]
    return imgs_array


class stage2net_preprocessor:

    def __init__(self, config, patch_size=32, overlap=None):

        self.patch_size = patch_size
        if not overlap:
            overlap = patch_size // 2
        self.step = patch_size - overlap
        self.outp1_path = config["segmentation_map_path"]
        # self.patients = config["training_patients"] + config["validation_patients"]
        # self.outp1 = outp1   # (155, 240, 240)
        self.result = {}

    def preprocessing(self, name):

        self.patient = name
        file_path = os.path.join(self.outp1_path, self.patient + ".nii.gz")
        self.outp1 = read_stik_to_nparray(file_path)
        best_patch_START = self.find_patch0()
        z_start, y_start, x_start = tuple(best_patch_START)

        return {"z": z_start, "y": y_start, "x": x_start, "size": self.patch_size}

    def find_patch0(self):
        """
        patch0 is the patch with highest_tumor_rate
        shape of (155, 240, 240)
        patient: x, y, z, w, h, d
        """
        orig_image = central_area_crop(self.outp1, crop_size=(128, 192, 160))
        array_shape = np.array(orig_image.shape)   #  (128, 192, 160)
        patch_shape = np.array([self.patch_size] * 3)   # (128)
        space = np.array([16] * 2, dtype=np.uint8)       # (8)
        patch_idx_limit = (array_shape[1:] - patch_shape[1:]) // space    # (4, 2)
        # construct an array, then np.argmax()
        patches_array = np.zeros(patch_idx_limit)
        for patch_idx_y in range(patch_idx_limit[0]):
            for patch_idx_x in range(patch_idx_limit[1]):
                patch_idx = np.array([patch_idx_y, patch_idx_x])
                patch_start = space * patch_idx
                patch_end = space * patch_idx + np.array(patch_shape[1:])
                cropped_array = orig_image[:, patch_start[0]:patch_end[0], patch_start[1]:patch_end[1]]
                num_tumor_voxel = (cropped_array > 0).sum()

                patches_array[patch_idx_y, patch_idx_x] = num_tumor_voxel
        argsmax = np.argwhere(patches_array == patches_array.max())
        patch_idx = argsmax[np.random.randint(len(argsmax))]
        # best_patch_idx = np.unravel_index(patches_array.argmax(), patches_array.shape)

        # convert in coords in the whole image
        orig_shape = np.array([155, 240, 240])
        cur_shape = np.array([128, 192, 160])
        coord_diffs = (orig_shape - cur_shape) // 2
        patch0_START_pt = np.array((0, ) + tuple(patch_idx * space)) + coord_diffs
        return patch0_START_pt

    def search_qualified_patch_surrounding(self):
        """
        :return:
        """
        step = self.step
        patch_shape = np.array([self.patch_size] * 3)
        # convert to coords in the whole image
        orig_shape = np.array([155, 240, 240])
        cur_shape = np.array([144, 192, 160])
        coord_diffs = (orig_shape - cur_shape) // 2
        best_patch_START = self.best_patch_start_coord + coord_diffs
        best_patch_END = best_patch_START + patch_shape

        # [6 orientations; for each orientation, determine whether +/- 16 surpass the border.]
        # for up and down, determine whether +/- 16 surpass the border (155)
        z_start, z_end = best_patch_START[0], best_patch_END[0]
        y_start, y_end = best_patch_START[1], best_patch_END[1]
        x_start, x_end = best_patch_START[2], best_patch_END[2]

        z_prev = 0 if (z_start - step < 0) else best_patch_START[0] - step
        z_next = 155 - self.patch_size if (z_end + step >= 155) else best_patch_START[0] + step
        x_prev = 0 if (x_start + step < 0) else best_patch_START[2] - step
        x_next = 240 - self.patch_size if (x_end + step >= 240) else best_patch_START[2] + step
        y_prev = 0 if (y_start + step < 0) else best_patch_START[1] - step
        y_next = 240 - self.patch_size if (y_end + step >= 240) else best_patch_START[1] + step   ## bug here, fixed


        z_cur, y_cur, x_cur = tuple(best_patch_START)
        qualified_patches = []
        for x_start in [x_prev, x_cur, x_next]:
            for y_start in [y_prev, y_cur, y_next]:
                for z_start in [z_prev, z_cur, z_next]:
                    cur_patch = self.outp1[z_start: z_start+self.patch_size, y_start: y_start+self.patch_size,
                                x_start: x_start+self.patch_size]
                    if ((self.patch_size == 32) and (cur_patch.sum() != 0)) or \
                            ((self.patch_size == 64) and ((cur_patch > 0).mean() >= 0.05)):
                        # save to txt:
                        qualified_patches.append({"z": z_start, "y": y_start, "x": x_start, "size": self.patch_size})
                    else:
                        continue

        self.result[self.patient] = qualified_patches  # {}






