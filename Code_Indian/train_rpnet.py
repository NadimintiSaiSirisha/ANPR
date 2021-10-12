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
    def __init__(self, num_points, num_classes, wrPath=None):
        super(fh02, self).__init__()
        self.load_wR2(wrPath)
        self.classifier1 = nn.Sequential(
            # nn.Dropout(),
            #nn.Linear(53248, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(53248, 4096),
            nn.Linear(4096,128),
            nn.Linear(128, statesNum),
        )
        self.classifier2 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(53248, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, digitsNum),
        )
        self.classifier3 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(53248, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, digitsNum),
        )
        self.classifier4 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(53248, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, alphaNum),
        )
        self.classifier5 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(53248, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, alphaNum),
        )
        self.classifier6 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(53248, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, digitsNum),
        )
        self.classifier7 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(53248, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, digitsNum),
        )

        self.classifier8 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(53248, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, digitsNum),
        )

        self.classifier9 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(53248, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, digitsNum),
        )




    def load_wR2(self, path):
        self.wR2 = wR2(numPoints)
        self.wR2 = torch.nn.DataParallel(self.wR2, device_ids=range(torch.cuda.device_count()))
        if not path is None:
            self.wR2.load_state_dict(torch.load(path))
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
        print("##################################################boxLoc that came from wR2 module classifier: ", boxLoc)
        h1, w1 = _x1.data.size()[2], _x1.data.size()[3]
        p1 = torch.FloatTensor([[w1, 0, 0, 0], [0, h1, 0, 0], [0, 0, w1, 0], [0, 0, 0, h1]]).cuda()
        p1.requires_grad = False
        h2, w2 = _x3.data.size()[2], _x3.data.size()[3]
        p2 = torch.FloatTensor([[w2, 0, 0, 0], [0, h2, 0, 0], [0, 0, w2, 0], [0, 0, 0, h2]]).cuda()
        p2.requires_grad = False
        h3, w3 = _x5.data.size()[2], _x5.data.size()[3]
        p3 = torch.FloatTensor([[w3, 0, 0, 0], [0, h3, 0, 0], [0, 0, w3, 0], [0, 0, 0, h3]]).cuda()
        p3.requires_grad = False

        # x, y, w, h --> x1, y1, x2, y2
        assert boxLoc.data.size()[1] == 4
        postfix = torch.FloatTensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]]).cuda()
        postfix.requires_grad = False
        boxNew = boxLoc.mm(postfix).clamp(min=0, max=1)

        # input = torch.rand(2, 1, 10, 10)
        # rois = torch.LongTensor([[0, 1, 2, 7, 8], [0, 3, 3, 8, 8], [1, 3, 3, 8, 8]])

        #print("_x1", _x1)
        #print("_x1.size():" , _x1.size())
        # _x1.size(): torch.Size([2, 64, 122, 122])
        roi1 = roi_pooling_ims(_x1, boxNew.mm(p1), size=(16, 8))
        roi2 = roi_pooling_ims(_x3, boxNew.mm(p2), size=(16, 8))
        roi3 = roi_pooling_ims(_x5, boxNew.mm(p3), size=(16, 8))
        rois = torch.cat((roi1, roi2, roi3), 1)

        _rois = rois.view(rois.size(0), -1)

        y0 = self.classifier1(_rois)
        y1 = self.classifier2(_rois)
        y2 = self.classifier3(_rois)
        y3 = self.classifier4(_rois)
        y4 = self.classifier5(_rois)
        y5 = self.classifier6(_rois)
        y6 = self.classifier7(_rois)
        y7 = self.classifier8(_rois)
        y8 = self.classifier9(_rois)
        
        return boxLoc, [y0, y1, y2, y3, y4, y5, y6, y7, y8]



def isEqual(labelGT, labelP):
    compare = [1 if int(labelGT[i]) == int(labelP[i]) else 0 for i in range(9)]
    # print(sum(compare))
    return sum(compare)


def eval(model, test_dirs, testAnnotationsPth):
    count, error, correct = 0, 0, 0
    dst = labelTestDataLoader(test_dirs, imgSize, testAnnotationsPth)
    # Make num_workers as 4 if paging error
    testloader = DataLoader(dst, batch_size=1, shuffle=True, num_workers=1)
    start = time()
    for i, (XI, labels, ims) in enumerate(testloader):

        count += 1
        YI = [[int(ee) for ee in el.split('_')[:9]] for el in labels]

        if use_gpu:
            x = XI.cuda(0)
        else:
            x = XI
        # Forward pass: Compute predicted y by passing x to the model

        fps_pred, y_pred = model(x)

        outputY = [el.data.cpu().numpy().tolist() for el in y_pred]
        labelPred = [t[0].index(max(t[0])) for t in outputY]

        #   compare YI, outputY
        try:
            if isEqual(labelPred, YI[0]) == 9:
                correct += 1
            else:
                pass
        except:
            error += 1
    return count, correct, error, float(correct) / count, (time() - start) / count



#Training model
def train_model(model, criterion, optimizer,testAnnotations, num_epochs=25):
    for epoch in range(epoch_start, num_epochs):
        print("Starting epoch: ", epoch)
        lossAver = []
        model.train(True)
        # Adding this line from CCPD master
        lrScheduler.step()
        start = time()

        for i, (XI, Y, labels, ims) in enumerate(trainloader):
            print("Training image: ", ims)
            #print("labels in train_model: ",labels)
            # labels in train_model:  ('12_24_29_8_7_26_29_24_25', '22_24_28_3_20_32_27_29_25')
            #print("type labels in train model: ", type(labels))
            # type labels in train model:  <class 'tuple'>
            #print("Training image ", i)
            if not len(XI) == batchSize:
                print(len(XI) , "is not equal to", batchSize)
                continue
            # labels is the corresponding license plate number (0_0_22_27_27_33_16)
            YI = [[int(ee) for ee in el.split('_')[:9]] for el in labels]
            
            #print("YI: ", YI)
            # YI:  [[12, 24, 29, 8, 7, 26, 29, 24, 25], [22, 24, 28, 3, 20, 32, 27, 29, 25]]
            #print("type(YI): ", type(YI))
            # type(YI):  <class 'list'>
            
            Y = np.array([el.numpy() for el in Y]).T  # Real value [cen_x,cen_y,w,h]
            print("Real value of boxLoc ", Y)

#Y:  [[0.50123762 0.35208333 0.27970297 0.04027778]
# [0.508      0.52419355 0.264      0.08602151]]
            
            #print("type(Y): ", type(Y))
            # type(Y):  <class 'numpy.ndarray'>
            
            if use_gpu:
                x = XI.cuda(0)
                # Adding requires_grad = False from CCPD_Master
                y = torch.FloatTensor(Y).cuda(0)
                y.requires_grad=False
            else:
                x = XI
                # Adding requires_grad = False from CCPD_Master
                y = torch.FloatTensor(Y)
                y.requires_grad=False
            # Forward pass: Compute predicted y by passing x to the model

            try:
                fps_pred, y_pred = model(x)  # fps_pred is the predicted [px,py,ph,pw], y_pred is the predicted value of the 7-digit license plate number
               

            except:
                continue

            # Compute and print loss
            loss = 0.0
            loss += 0.8 * nn.L1Loss().cuda()(fps_pred[:,:2], y[:,:2])  # Locate cen_x and cen_y losses
            
            #tensor(0.0233, device='cuda:0', grad_fn=<AddBackward0>)
            loss += 0.2 * nn.L1Loss().cuda()(fps_pred[:,2:], y[:,2:]) # Locate w and h loss
            print("loss due to bounding box: #############################", loss)
            # tensor(0.0282, device='cuda:0', grad_fn=<AddBackward0>)
            for j in range(9):  #Cross entropy loss for each number plate
                l = torch.LongTensor([el[j] for el in YI]).cuda(0)
                loss += criterion(y_pred[j], l) # Classification loss

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Commenting the below line as per CCPD_Master
            #lrScheduler.step()

            try:
                lossAver.append(loss.item())
            except:
                pass

            #modelFolder = FolderName/
            if i % 50 == 1:
                with open(writeFile, 'a') as outF:
                    outF.write('train %s images, use %s seconds, loss %s\n' % (
                        i * batchSize, time() - start,
                        sum(lossAver) / len(lossAver) if len(lossAver) > 0 else 'NoLoss'))
                storeName = modelFolder+"epoch"+str(epoch)+"image"+str(i)+".pth"
                print("Saving the dict file................................... here", storeName)
                #torch.save(model.state_dict(), storeName)
                print('%s %s %s\n' % (epoch, sum(lossAver) / len(lossAver), time() - start))
        model.eval()
        count, correct, error, precision, avgTime = eval(model, testDirs, testAnnotations)
        with open(writeFile, 'a') as outF:
            outF.write('%s %s %s\n' % (epoch, sum(lossAver) / len(lossAver), time() - start))
            outF.write('*** total %s error %s precision %s avgTime %s\n' % (count, error, precision, avgTime))
        print('*** total %s error %s precision %s avgTime %s\n' % (count, error, precision, avgTime))
        storeName = modelFolder+"epoch"+str(epoch)+".pth"
        print("Saving the dict file................................... here",  storeName)
        torch.save(model.state_dict(), storeName)
    return model


if __name__ == '__main__':

    wR2Path = './wR2FineTune/wR2epoch_299.pth'

    images = './proper_dataset/rpnet10/image/train'

    trainAnnotations = './proper_dataset/rpnet10/label'

    epochs = 100
    batch_size = 1
    start_epoch =0
    test = './proper_dataset/rpnet10/image/val'

    testAnnotations = './proper_dataset/rpnet10/label'
  
    storeModel = './proper_dataset/rpnet_layer'

    writeFile = './proper_dataset/rpnet_layer/fh02_6_layer.out'
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("CUDA is available. Using CUDA...")
    else:
        print("CUDA is not available. This may result in some errors...")
        
    numClasses = 9  # The number of license plates is 9
    numPoints = 4  # The number of positioning points is 4
    
    imgSize = (480, 480)  # The picture size is 480*480
    #provNum, alphaNum, adNum = 38, 25, 35  # The number of province categories, the number of urban areas, the number of characters
    statesNum, digitsNum, alphaNum = 38, 10, 24
    batchSize = batch_size if use_gpu else 2
    trainDirs = images.split(',')
    testDirs = test.split(',')
    modelFolder = storeModel if storeModel[-1] == '/' else storeModel + '/'
    if not os.path.isdir(modelFolder):
        os.mkdir(modelFolder)

    #   initialize the output file
    if not os.path.isfile(writeFile):
        with open(writeFile, 'wb') as outF:
            pass
    resume ='111'
    resume_file = resume
    epoch_start = 0
    
    #resume_file = resume
    
    if not resume_file == '111':  # Retraining
        # epoch_start = int(resume_file[resume_file.find('pth') + 3:]) + 1
        if not os.path.isfile(resume_file):
            print("fail to load existed model! Existing ...")
            exit(0)
        print("Load existed model! %s" % resume_file)
        model_conv = fh02(numPoints, numClasses)
        model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
        model_conv.load_state_dict(torch.load(resume_file))
        model_conv = model_conv.cuda()
    else:
        model_conv = fh02(numPoints, numClasses, wR2Path)
        if use_gpu:
            model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
            model_conv = model_conv.cuda()

    #print(model_conv)
    print('Model parameters' + str(get_n_params(model_conv)))
    # Model parameters68716914
    criterion = nn.CrossEntropyLoss()
    # optimizer_conv = optim.RMSprop(model_conv.parameters(), lr=0.01, momentum=0.9)
    # optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)
    optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)
    dst = labelFpsDataLoader(trainDirs, imgSize, trainAnnotations)
    trainloader = DataLoader(dst, batch_size=batchSize, shuffle=True, num_workers=1)
    lrScheduler = lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.1) # Every 5 epochs, the learning rate is multiplied by 0.1
    model_conv = train_model(model_conv, criterion, optimizer_conv, testAnnotations, num_epochs=epochs)