

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

import os
import torch
import numpy as np


def cal_mIoU(label, pred, num_classes):

    label, pred = label.cpu().numpy(), pred.cpu().numpy()

    print(label.shape)
    iou_list = []
    iou_sum = 0
    for i in range(num_classes):
        son = np.sum((label == i) & (pred == i))
        mom = np.sum(label == i) + np.sum(pred == i) - son

        # Exception
        IoU = son / mom if mom != 0 else float('nan')
        iou_sum += IoU if mom != 0 else 0

        iou_list.append(IoU)

    mIoU = iou_sum / num_classes
    return mIoU, iou_list


def calc_miou(pred_batch, label_batch, num_classes, ignore_index):
    '''
    :param pred_batch: [b,h,w]
    :param label_batch: [b,h,w]
    :param num_classes: scalar
    :param ignore_index: scalar
    :return:
    '''
    miou_sum, miou_count = 0, 0
    for batch_idx in range(label_batch.shape[0]):
        pred, label = pred_batch[batch_idx].flatten(0), label_batch[batch_idx].flatten(0)

        mask = label != ignore_index
        pred, label = pred[mask], label[mask]

        pred_one_hot = torch.nn.functional.one_hot(pred, num_classes)
        label_one_hot = torch.nn.functional.one_hot(label, num_classes)

        intersection = torch.sum(pred_one_hot * label_one_hot)
        union = torch.sum(pred_one_hot) + torch.sum(label_one_hot) - intersection + 1e-6

        miou_sum += intersection / union
        miou_count += 1
    return miou_sum / miou_count



