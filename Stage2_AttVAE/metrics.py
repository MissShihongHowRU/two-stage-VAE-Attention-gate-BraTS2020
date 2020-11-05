"""
@author: Chenggang
@github: https://github.com/MissShihongHowRU
@time: 2020-09-09 22:04
"""
import torch
from torch.nn.modules.loss import _Loss 
import torch.nn as nn

class FocalLoss(_Loss):
    '''
    Focal_Loss = - [alpha * (1 - p)^gamma *log(p)]  if y = 1;
               = - [(1-alpha) * p^gamma *log(1-p)]  if y = 0;
        average over batchsize; alpha helps offset class imbalance; gamma helps focus on hard samples
    '''
    def __init__(self, alpha=0.9, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true, eps=1e-8):

        alpha = self.alpha
        gamma = self.gamma
        focal_ce = - (alpha * torch.pow((1-y_pred), gamma) * torch.log(torch.clamp(y_pred, eps, 1.0)) * y_true
                      + (1-alpha) * torch.pow(y_pred, gamma) * torch.log(torch.clamp(1-y_pred, eps, 1.0)) * (1-y_true))
        focal_loss = torch.mean(focal_ce)

        return focal_loss

class SoftDiceLoss(_Loss):
    '''
    Soft_Dice = 2*|dot(A, B)| / (|dot(A, A)| + |dot(B, B)| + eps)
    eps is a small constant to avoid zero division,
    '''
    def __init__(self, combine):
        super(SoftDiceLoss, self).__init__()
        self.combine = combine

    def forward(self, y_pred, y_true, eps=1e-8):   # put 2,1,4 together

        if self.combine:
            y_pred[:, 0, :, :, :] = torch.sum(y_pred, dim=1)
            y_pred[:, 1, :, :, :] = torch.sum(y_pred[:, 1:, :, :, :], dim=1)
            y_true[:, 0, :, :, :] = torch.sum(y_true, dim=1)
            y_true[:, 1, :, :, :] = torch.sum(y_true[:, 1:, :, :, :], dim=1)

        intersection = torch.sum(torch.mul(y_pred, y_true), dim=[-3, -2, -1])
        union = torch.sum(torch.mul(y_pred, y_pred),
                          dim=[-3, -2, -1]) + torch.sum(torch.mul(y_true, y_true), dim=[-3, -2, -1]) + eps

        dice = 2 * intersection / union   # (bs, 3)
        # dice_loss = 1 - torch.mean(dice)  # loss small, better
        means = torch.mean(dice, dim=0)
        dice_loss = 1 - 0.1*means[0] - 0.45*means[1] - 0.45*means[2]

        return dice_loss


class CustomKLLoss(_Loss):
    '''
    KL_Loss = (|dot(mean , mean)| + |dot(std, std)| - |log(dot(std, std))| - 1) / N
    N is the total number of image voxels
    '''
    def __init__(self, *args, **kwargs):
        super(CustomKLLoss, self).__init__()

    def forward(self, mean, std):
        return torch.mean(torch.mul(mean, mean)) + torch.mean(torch.mul(std, std)) - torch.mean(torch.log(torch.mul(std, std))) - 1


class CombinedLoss(_Loss):
    '''
    Combined_loss = Dice_loss + k1 * L2_loss + k2 * KL_loss
    As default: k1=0.1, k2=0.1
    '''
    def __init__(self, combine=False, k1=0.1, k2=0.1):
        super(CombinedLoss, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.dice_loss = SoftDiceLoss(combine)
        self.l2_loss = nn.MSELoss()
        self.kl_loss = CustomKLLoss()

    def forward(self, y_pred, y_true, y_mid):
        est_mean, est_std = (y_mid[:, :128], y_mid[:, 128:])
        seg_pred, seg_truth = (y_pred[:,:3,:,:,:], y_true[:,:3,:,:,:])   # 10; (3+4)
        vae_pred, vae_truth = (y_pred[:,3:,:,:,:], y_true[:,3:,:,:,:])
        dice_loss = self.dice_loss(seg_pred, seg_truth)
        l2_loss = self.l2_loss(vae_pred, vae_truth)  ### 7; 4
        kl_div = self.kl_loss(est_mean, est_std)

        combined_loss = dice_loss + self.k1 * l2_loss + self.k2 * kl_div

        return combined_loss
