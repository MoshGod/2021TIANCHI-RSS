
import random
import torch
import numpy as np
import pandas as pd
import pathlib, sys, os, time
import numba, cv2, gc
import glob
import torchvision
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import albumentations as A
import torch.nn.functional as F
import torch.utils.data as D

from torchvision import transforms as T
from pytorch_toolbelt import losses as L
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm_notebook
from PIL import Image
from torch import nn, optim

torch.backends.cudnn.enabled = True

from model.FPN import FPN
from model.UnetPP import UnetPP
from model.PSPNet import PSPNet


''' Set random seed'''
seed = 6666  
if seed is not None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


''' Calculate mIoU '''
def cal_mIoU(label, pred, num_classes=10):

    label, pred = label.cpu().numpy(), pred.cpu().numpy()

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
    return mIoU, np.array(iou_list, dtype=np.float32)


''' Dataset '''
class RSSDataset(D.Dataset):
    def __init__(self, paths, transform, test_mode=False):
        self.paths = paths
        self.transform = transform
        self.test_mode = test_mode
        
        self.len = len(paths)
        self.as_tensor = T.Compose([
            T.ToTensor()
        ])
        
    # 4 channels
    def __getitem__(self, index):
        img = Image.open(self.paths[index])
        img = np.array(img)
        if not self.test_mode:
            mask = Image.open(self.paths[index].replace('.tif', '.png'))
            mask = np.array(mask) - 1
            transformed = self.transform(image=img, mask=mask)
#             return self.as_tensor(img), torch.from_numpy(mask).long()
            return self.as_tensor(transformed['image']), torch.from_numpy(transformed['mask']).long()
#         else:
#             return self.as_tensor(img), ''
    
    
    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len


''' Data Augmentation '''
trfm = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.OneOf([
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
    ]),
#     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#     ToTensorV2(),
])

dataset = RSSDataset(
    glob.glob('../tcdata/suichang_round1_train_210120/*.tif'),
    trfm, False
)


''' Visualization '''
CLASSES = ["耕地", "林地", "草地", "道路", "城镇建设用地", "农村建设用地", "工业用地", "构筑物", "水域", "裸地"]
COLORMAP = [
    [255,204,0],
    [0,128,0],
    [51,204,51],
    [221,221,221],
    [255,204,204],
    [204,255,204],
    [153,153,255],
    [255,255,204],
    [51,204,255],
    [0, 0, 0]]
colormap2label = torch.zeros(256**3, dtype=torch.uint8)

for i, colormap in enumerate(COLORMAP):
    colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i

def label2image(pred):
    colormap = torch.tensor(COLORMAP, device='cuda', dtype=torch.int)
    x = pred.long()
    return (colormap[x,:]).data.cpu().numpy()

def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    plt.show()
    return axes

def plot_curve(data, y_legend='value', x_legend='batch', title=None): # draw loss
    fig = plt.figure()
    plt.plot(range(len(data)), data, color='blue')
    if title:
        plt.title(title)
    plt.legend(['value'], loc='upper right')
    plt.xlabel(x_legend)
    plt.ylabel(y_legend)
    plt.show()


''' Model parameters '''
IMAGE_SIZE = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
model = UnetPP('efficientnet-b6', 4, 10).to(DEVICE)
model_name = 'UnetPP-b6'
# model = UnetPP('efficientnet-b7', 4, 10).to(DEVICE)
# model_name = 'UnetPP-b7'
# model = FPN('efficientnet-b6', 4, 10).to(DEVICE)
# model_name = 'FPN-b6'
# model = FPN('efficientnet-b7', 4, 10).to(DEVICE)
# model_name = 'FPN-b7'
# model = PSPNet('efficientnet-b6', 4, 10).to(DEVICE)
# model_name = 'PSPNet-b6'
EPOCHES = 60
BATCH_SIZE = 8
lr = 3e-4
snapshot = 6
scheduler_step = EPOCHES / snapshot
min_lr = 0.00005
momentum = 0.9
weight_decay = 5e-4

optimizer = optim.AdamW(model.parameters(), lr=lr ,weight_decay=weight_decay)
# criterion = nn.CrossEntropyLoss().to(DEVICE)
criterion = L.JointLoss(L.DiceLoss(mode="multiclass"), L.SoftCrossEntropyLoss(smooth_factor=0.1), 0.5, 0.5).to(DEVICE)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.6)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1, eta_min=min_lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-5, last_epoch=-1)


''' Validation '''
@torch.no_grad()
def validation(model, loader, criterion, v=False):
    step, total, imgs, num_images, correct, validate_loss = 0, 0, [], 0, 0, 0.0
    miou, val_iou = 0.0, []
    was_training = model.training
    model.eval()
    vit = 0
    for image, target in loader:
        step += 1
        total += image.size(0)
        image, target = image.to(DEVICE), target.to(DEVICE)
        output = model(image)
        
        validate_loss += criterion(output, target).item()
        pred = output.argmax(1)
        correct += torch.eq(pred, target).sum().float().item() / (image.size(2)*image.size(3))
        each_iou, iou_list = cal_mIoU(pred, target)
        miou += each_iou
        
        val_iou.append(iou_list)
        
        if v and vit < 4:
            for i in range(image.size(0)):
                img_nd = image.permute(0, 2, 3, 1) * 255
                pred = torch.argmax(output.data, dim=1)
                pred1 = label2image(pred[i])
                imgs += [img_nd[i].data.int().cpu().numpy(), pred1, label2image(target[i])]
                num_images += 1
                if num_images == 4:
                    vit += 1
                    show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, 4)  # show only 4 images
                    num_images, imgs = 0, []
                    break
                    
    model.train(mode=was_training)
    return validate_loss/step, correct, total, miou/step, val_iou


''' Train '''
# model_path = 'model_file/SegQyl_54_0.5976_0.9079_0.3506.pth' 
# model.load_state_dict(torch.load(model_path))
model.train()

header = r'''
        Train | Valid
Epoch |  Loss |  IOU | Time, m'''

class_name = ['farm','land','forest','grass','road','urban_area',
                 'countryside','industrial_land','construction',
                 'water', 'bareland']

#          Epoch         metrics            time
raw_line = '{:6d}' + '\u2502{:7.3f}'*2 + '\u2502{:6.2f}'

for fold_idx in range(1, 6): # k-fold
#     valid_idx, train_idx = [], []
#     for i in range(len(dataset)):
#         if (fold_idx + i) % 7 == 0:
#             valid_idx.append(fold_idx + i)
#         else:
#             train_idx.append(i)

#     train_ds = D.Subset(dataset, train_idx)
#     valid_ds = D.Subset(dataset, valid_idx)
    loader = D.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=(0 if sys.platform.startswith('win32') else 4))
#     vloader = D.DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=(0 if sys.platform.startswith('win32') else 4))

    best_iou = 0.4
    train_loss = []
#     train_iou = []
    for epoch in range(1, EPOCHES+1):
        print('\n',' -'*5,'[Epoch] {}/{}'.format(epoch, EPOCHES),'- ' * 25)
        total, epoch_loss, epoch_correct, batch_idx = 0, 0.0, 0.0, 0
        
        start_time = time.time()
        for image, target in tqdm_notebook(loader):
            batch_idx += 1
            total += image.size(0)
            image, target = image.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * image.size(0)
            epoch_correct += torch.sum(torch.argmax(output.data, 1) == target.data) / (image.size(2)*image.size(3))
            if batch_idx % 10 == 0:
                train_loss.append(epoch_loss/total)

            if batch_idx % 100 == 0:
                # print('-' * 10)
                # for param in model.parameters():
                #     print('参数范围', torch.min(param.data), torch.max(param.data))
                #     break
                # print('输出范围', torch.min(logits.data), torch.max(logits.data))
                # print('类别范围', torch.min(torch.argmax(logits.data, dim=1)), torch.max(torch.argmax(logits.data, dim=1)))

                # _, _, _, miou, _ = validation(model, vloader, criterion)
                # train_iou.append(miou)  # miou            
                print('[Batch {}] Loss {:.4f}  '.format( batch_idx, loss.item()),
                    '[Train] Loss {:.4f}  Acc {:.4f}'.format(epoch_loss / total, epoch_correct.double() / total)) 
            
        if scheduler:
            scheduler.step()
            
        plot_curve(train_loss, y_legend='loss', title='train loss epoch '+str(epoch)) # loss曲线
#         plot_curve(train_iou, y_legend='miou', title='val_iou epoch '+str(epoch)) # iou曲线
        
#         vloss, vcorrest, vtotal, viou, viou_list = validation(model, vloader, criterion, v=True)
#         vacc = vcorrest/vtotal
        tlr = optimizer.state_dict()['param_groups'][0]['lr']
        print('[epoch lr] {}'.format(tlr))
#         print('[Validation] Loss {:.4f}  Acc {}/{}={:.4f}'.format(vloss, round(vcorrest), vtotal, vacc))
#         print(header)
#         print(raw_line.format(epoch, vloss, viou, (time.time()-start_time)/60**1))
#         print('  '.join(class_name))
#         print('\t'.join(np.nanmean(np.stack(viou_list), 0).round(3).astype(str)))
        print()
        if epoch>=40 and epoch <=48:
#             best_iou = viou
            torch.save(model.state_dict(), '../user_data/model_data/{}_{}_{:.6f}.pth'.format(model_name, epoch, tlr))
    break