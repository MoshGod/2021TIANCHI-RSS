
import torch
import glob
import numpy as np
import sys, os
import cv2
import torchvision
import warnings
warnings.filterwarnings('ignore')
import torch.nn.functional as F
import torch.utils.data as D
import albumentations as A
import matplotlib.pyplot as plt


from tqdm import tqdm_notebook
from PIL import Image
from torch import nn, optim
from torchvision import transforms as T
from torch.optim.swa_utils import AveragedModel
from model.FPN import FPN
from model.UnetPP import UnetPP
from model.PSPNet import PSPNet

torch.backends.cudnn.enabled = True


IMAGE_SIZE = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# model_psp_b6 = PSPNet('efficientnet-b6', 4, 10).to(DEVICE)
# model_psp_b6 = AveragedModel(model_psp_b6)
# model_psp_b6.load_state_dict(torch.load('../user_data/model_data/pspnet-b6_swa60_stretch.pth'))
# model_psp_b6.eval()

model_fpn_b6 = FPN('efficientnet-b6', 4, 10).to(DEVICE)
model_fpn_b6.load_state_dict(torch.load('../user_data/model_data/FPN-b6.pth'))
model_fpn_b6.eval()

model_fpn_b7 = FPN('efficientnet-b7', 4, 10).to(DEVICE)
model_fpn_b7.load_state_dict(torch.load('../user_data/model_data/FPN-b7.pth'))
model_fpn_b7.eval()

model_unetpp_b6 = UnetPP('efficientnet-b6', 4, 10).to(DEVICE)
model_unetpp_b6.load_state_dict(torch.load('../user_data/model_data/UnetPP-b6.pth'))
model_unetpp_b6.eval()

model_unetpp_b7 = UnetPP('efficientnet-b7', 4, 10).to(DEVICE)
model_unetpp_b7.load_state_dict(torch.load('../user_data/model_data/UnetPP-b7.pth'))
model_unetpp_b7.eval()



trfm = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.ToTensor(),
])

for idx, name in enumerate(tqdm_notebook(glob.glob('../tcdata/suichang_round2_test_partB_210316/*.tif')[:])):
    image = Image.open(name)
    image = trfm(image)
    with torch.no_grad():
        image = image.to(DEVICE)[None]
        image_flip2 = torch.flip(image, [0, 3])
        image_flip3 = torch.flip(image, [0, 2])

        output1 = model_fpn_b7(image).cpu().numpy()
        output2 = model_unetpp_b6(image).cpu().numpy()
        output3 = model_fpn_b6(image).cpu().numpy()
        output10 = model_unetpp_b7(image).cpu().numpy()

        output4 = torch.flip(model_fpn_b7(image_flip2), [3, 0]).cpu().numpy()
        output5 = torch.flip(model_unetpp_b6(image_flip2), [3, 0]).cpu().numpy()
        output6 = torch.flip(model_fpn_b6(image_flip2), [3, 0]).cpu().numpy()
        output11 = torch.flip(model_unetpp_b7(image_flip2), [3, 0]).cpu().numpy()

        output7 = torch.flip(model_fpn_b7(image_flip3), [2, 0]).cpu().numpy()
        output8 = torch.flip(model_unetpp_b6(image_flip3), [2, 0]).cpu().numpy()
        output9 = torch.flip(model_fpn_b6(image_flip3), [2, 0]).cpu().numpy()
        output12 = torch.flip(model_unetpp_b7(image_flip3), [2, 0]).cpu().numpy()

        score = (output1 + output2 + output3 + output4 + output5 +
                 output6 + output7 + output8 + output9 + output10 +
                 output11 + output12) / 12.0
        score_sigmoid = score[0].argmax(0) + 1

        print(score_sigmoid.min(), score_sigmoid.max())
        cv2.imwrite('../prediction_result/' + name.split('/')[-1].replace('.tif', '.png'), score_sigmoid)
