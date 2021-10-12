import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
import argparse
from time import time
from load_data import *
from torch.optim import lr_scheduler

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

class wR2(nn.Module):
  # Doubt 1: WHy is num_classes taken as 1000
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

def train_model(model, criterion, optimizer, num_epochs=25):
    # since = time.time()
    for epoch in range(epoch_start, num_epochs):
        lossAver = []
        model.train(True)
        lrScheduler.step()
        start = time()

        for i, (XI, YI) in enumerate(trainloader):
            # print('%s/%s %s' % (i, times, time()-start))
            YI = np.array([el.numpy() for el in YI]).T
            if use_gpu:
                x = Variable(XI.cuda(0))
                y = Variable(torch.FloatTensor(YI).cuda(0), requires_grad=False)
            else:
                x = Variable(XI)
                y = Variable(torch.FloatTensor(YI), requires_grad=False)
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(x)

            # Compute and print loss
            loss = 0.0
            if len(y_pred) == batchSize:
                loss += 0.8 * nn.L1Loss().cuda()(y_pred[:,:2], y[:,:2])
                loss += 0.2 * nn.L1Loss().cuda()(y_pred[:,2:], y[:,2:])
                lossAver.append(loss.item())

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                torch.save(model.state_dict(), storeName)
            if i % 50 == 1:
                with open(writeFileForOutput, 'a') as outF:
                    outF.write('train %s images, use %s seconds, loss %s\n' % (i*batchSize, time() - start, sum(lossAver[-50:]) / len(lossAver[-50:])))
        print ('%s %s %s\n' % (epoch, sum(lossAver) / len(lossAver), time()-start))
        with open(writeFileForOutput, 'a') as outF:
            outF.write('Epoch: %s %s %s\n' % (epoch, sum(lossAver) / len(lossAver), time()-start))
        torch.save(model.state_dict(), storeName + "epoch_"+str(epoch)+".pth")
    return model


if __name__=='__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=True,help="path to the input file")

    ap.add_argument("-n", "--epochs", default=10000, help="epochs for train")
    #images = "./One_image"
    #epochs = 25
    ap.add_argument("-b", "--batchsize", default=4, help="batch size for train")
    #batchsize = 1
    ap.add_argument("-w", "--writeFileForOutput", default='wR2.out', help="file for output")
    args = vars(ap.parse_args())
    
    
    images = str(args["images"])
    epochs = int(args["epochs"])
    batchsize = int(args["batchsize"])
    writeFileForOutput = str(args["writeFileForOutput"])
    use_gpu = torch.cuda.is_available()
    print(use_gpu)
    
    resumeFileForRetrain = "111"
    numClasses = 4
    imgSize = (480, 480)
    #batchSize = int(args["batchsize"]) if use_gpu else 8
    batchSize = batchsize if use_gpu else 2
    
    modelFolder = 'wR2/'
    storeName = modelFolder + 'wR2'

    # make the folder wR2 if it is not already present
    if not os.path.isdir(modelFolder):
        os.mkdir(modelFolder)

    #epochs = int(args["epochs"])
    # Open the wR2.out file
    with open(writeFileForOutput, 'wb') as outF:
        pass
    epoch_start = 0
    #resume_file = str(args["resume"])
    resume_file = resumeFileForRetrain
    if not resume_file == '111':#resume means to continue training from the saved model after the last training
        # epoch_start = int(resume_file[resume_file.find('pth') + 3:]) + 1
        if not os.path.isfile(resume_file):
            print("fail to load existed model! Existing ...")
            exit(0)
        print("Load existed model! %s" % resume_file)
        model_conv = wR2(numClasses)
        model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
        model_conv.load_state_dict(torch.load(resume_file))
        model_conv = model_conv.cuda()
    else:   #Training from scratch
        model_conv = wR2(numClasses)
        if use_gpu:
            model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
            model_conv = model_conv.cuda()

    print(model_conv)
    print(get_n_params(model_conv))
   
#Identification module uses cross entropy loss function
    criterion = nn.MSELoss()
  
#Optimization function
    optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)
    
#Adjust the learning rate during training. For every 5 epochs, the learning rate is multiplied by 0.1
    lrScheduler = lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.1)
    # optimizer_conv = optim.Adam(model_conv.parameters(), lr=0.01)
    # dst = LocDataLoader([args["images"]], imgSize)
    dst = ChaLocDataLoader(images.split(','), imgSize)
    trainloader = DataLoader(dst, batch_size=batchSize, shuffle=True, num_workers=4)

    model_conv = train_model(model_conv, criterion, optimizer_conv, num_epochs=epochs)
