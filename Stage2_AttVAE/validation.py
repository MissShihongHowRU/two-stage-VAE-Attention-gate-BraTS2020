"""
@author: Chenggang
@github: https://github.com/MissShihongHowRU
@time: 2020-09-09 22:04
"""
import torch
from torch.autograd import Variable
from tqdm import tqdm
import sys
sys.path.append(".")
from utils import AverageMeter, calculate_accuracy

def val_epoch(epoch, data_set, model, criterion, optimizer, opt, logger):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    losses = AverageMeter()
    WT_dice = AverageMeter()
    TC_dice = AverageMeter()
    ET_dice = AverageMeter()

    valildation_loader = torch.utils.data.DataLoader(dataset=data_set,
                                     batch_size=opt["validation_batch_size"], 
                                     shuffle=False,
                                     pin_memory=True)
    val_process = tqdm(valildation_loader)
    for i, (input_, targets) in enumerate(val_process):
        if i > 0:
            val_process.set_description("Epoch:%d;Loss:%.4f; WT:%.4f, TC:%.4f, ET:%.4f; lr: %.6f"%(epoch,
                                                        losses.avg.item(), WT_dice.avg.item(), TC_dice.avg.item(),
                                                        ET_dice.avg.item(), optimizer.param_groups[0]['lr']))

        if opt["cuda_devices"] is not None:
            input_ = input_.type(torch.FloatTensor)
            input_ = input_.cuda()
            targets = targets.type(torch.FloatTensor)
            targets = targets.cuda()

        with torch.no_grad():
            if opt["VAE_enable"]:
                outputs, distr = model(input_)
                loss = criterion(outputs, targets, distr)
            else:
                outputs = model(input_)
                loss = criterion(outputs, targets)

        acc = calculate_accuracy(outputs.cpu(), targets.cpu())

        losses.update(loss.cpu(), input_.size(0))
        WT_dice.update(acc["dice_wt"], input_.size(0))
        TC_dice.update(acc["dice_tc"], input_.size(0))
        ET_dice.update(acc["dice_et"], input_.size(0))

    logger.log(phase="val", values={
        'epoch': epoch,
        'loss': format(losses.avg.item(), '.4f'),
        'wt-dice': format(WT_dice.avg.item(), '.4f'),
        'tc-dice': format(TC_dice.avg.item(), '.4f'),
        'et-dice': format(ET_dice.avg.item(), '.4f'),
        'lr': optimizer.param_groups[0]['lr']
    })
    return losses.avg, WT_dice.avg, TC_dice.avg, ET_dice.avg