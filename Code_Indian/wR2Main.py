import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
#from visdom import Visdom
import argparse
from time import time
from load_data import *
from roi_pooling import roi_pooling_ims
from torch.optim import lr_scheduler
from CalculateIoU import * 
import cv2 

# Calculate the number of model parameters
def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

class wR2(nn.Module):
    def __init__(self, num_classes=1000):
        super(wR2, self).__init__()
        hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden9 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden10 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        self.features = nn.Sequential(
            hidden1,
            hidden2,
            hidden3,
            hidden4,
            hidden5,
            hidden6,
            hidden7,
            hidden8,
            hidden9,
            hidden10
        )
        self.classifier = nn.Sequential(
            nn.Linear(23232, 100),
            # nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            # nn.ReLU(inplace=True),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x1 = self.features(x)
        x11 = x1.view(x1.size(0), -1)
        x = self.classifier(x11)
        return x


class fh02(nn.Module):
    def __init__(self, num_points, wrPath=None):
        super(fh02, self).__init__()
        self.load_wR2(wrPath)

    def load_wR2(self, path):
        self.wR2 = wR2(numPoints)
        self.wR2 = torch.nn.DataParallel(self.wR2, device_ids=range(torch.cuda.device_count()))
        if not path is None:
            state_dict = torch.load(path)
            from collections import OrderedDict
            new_state_dict = OrderedDict()

            for k,v in state_dict.items():
                if 'module' not in k:
                    k = 'module.' + k
                else:
                    k = k.replace('features.module.', 'module.features.')
                new_state_dict[k]=v
            self.wR2.load_state_dict(new_state_dict)
            #self.wR2.load_state_dict(torch.load(path))
            # self.wR2 = self.wR2.cuda()
        # for param in self.wR2.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        x0 = self.wR2.module.features[0](x)
        _x1 = self.wR2.module.features[1](x0)
        x2 = self.wR2.module.features[2](_x1)
        _x3 = self.wR2.module.features[3](x2)
        x4 = self.wR2.module.features[4](_x3)
        _x5 = self.wR2.module.features[5](x4)
        x6 = self.wR2.module.features[6](_x5)
        x7 = self.wR2.module.features[7](x6)
        x8 = self.wR2.module.features[8](x7)
        x9 = self.wR2.module.features[9](x8)
        x9 = x9.view(x9.size(0), -1)
        boxLoc = self.wR2.module.classifier(x9)
        
        return boxLoc


if __name__ == '__main__':

#########################################
# If using colab, uncomment the following chunk of code:
    wR2Path = './wR2FineTune/wR2epoch_299.pth'
    images = './proper_dataset/v3andv4/shifted/image'
    annotationsFolder = './proper_dataset/v3andv4/shifted/annotation'
    testDir = images.split(',')
    dstrectangleDir = './proper_dataset/v3andv4/shifted/rectangle_finetuned_290921/'
    writeIousInFile = './proper_dataset/v3andv4/shifted/wR2finetuned_290921.out'
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("CUDA is available. Using CUDA...")
    else:
        print("CUDA is not available. This may result in some errors...")
        

    numPoints = 4  # The number of positioning points is 4
    
    imgSize = (480, 480)  # The picture size is 480*480
    # The number of states, rest characters

    batchSize = 1
    model_conv = fh02(numPoints, wR2Path)
    if use_gpu:
        model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
        model_conv = model_conv.cuda()
   
    dst = ChaLocDataLoader(testDir, imgSize,  annotationsFolder)
    testloader = DataLoader(dst, batch_size=batchSize, shuffle=False, num_workers=1)
    sumIOU = 0
    totalImgs = 0
    for i, (XI, bbratio, ims) in enumerate(testloader):

            if use_gpu:
                x = XI.cuda(0)
            else:
                x = XI
            
            cv2Img = cv2.imread(ims[0])
            [acx, acy, aw, ah ] = bbratio
            a_left_up =  [(acx - aw / 2) * cv2Img.shape[1], (acy - ah / 2) * cv2Img.shape[0]]
            a_right_down = [(acx + aw / 2) * cv2Img.shape[1], (acy + ah / 2) * cv2Img.shape[0]]
            gt_box = [int(a_left_up[0]), int(a_left_up[1]), int(a_right_down[0]), int(a_right_down[1])]

           

            fps_pred = model_conv(x)  # fps_pred is the predicted [px,py,ph,pw], y_pred is the predicted value of the 7-digit license plate number
            [cx, cy, w, h] = fps_pred.data.cpu().numpy()[0].tolist()
            cv2Img = cv2.imread(ims[0])
            #cv2_imshow(cv2Img)
            #cv2.waitKey(0)
            left_up = [(cx - w / 2) * cv2Img.shape[1], (cy - h / 2) * cv2Img.shape[0]]
            #print("left_up:", left_up)
            right_down = [(cx + w / 2) * cv2Img.shape[1], (cy + h / 2) * cv2Img.shape[0]]
            #print("right_down:", right_down)
            pred_box = [int(left_up[0]), int(left_up[1]), int(right_down[0]), int(right_down[1])]
            cv2.rectangle(cv2Img, (int(a_left_up[0]), int(a_left_up[1])), (int(a_right_down[0]), int(a_right_down[1])), (0, 255, 0),2)
            cv2.rectangle(cv2Img, (int(left_up[0]), int(left_up[1])), (int(right_down[0]), int(right_down[1])), (0, 0, 255),2)
            imgname = ims[0].split( "/")[-1]
            #cv2.imshow(imgname, cv2Img)
            #cv2.waitKey(0)
            imgname = ims[0].split( '\\')[-1]
            print("Only image name: ", imgname)
            dstFilePath = dstrectangleDir + imgname
            #print('Picture save address', dstFilePath)
            iou = calc_iou(gt_box,pred_box )
            print("IoU for image: ", imgname ," - ", iou)
            if(iou>0.5):
                print("imagename: ", ims[0])
                cv2ImgOriginal = cv2.imread(ims[0])
                cv2.imwrite('./proper_dataset/rpnet/'+imgname, cv2ImgOriginal )
            totalImgs+=1
            sumIOU+= iou
            with open(writeIousInFile, 'a') as outF:
                    outF.write('Image name: '+ imgname+ 'IoU: '+ str(iou)+ '\n')
            cv2.imwrite(dstFilePath, cv2Img)
        
    print("Average IOU: ", sumIOU/totalImgs) 
    with open(writeIousInFile, 'a') as outF:
        outF.write("Average IOU: "+ str(sumIOU/totalImgs) + "/n")