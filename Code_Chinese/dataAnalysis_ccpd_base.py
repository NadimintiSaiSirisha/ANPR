
from torch.utils.data import BatchSampler,Dataset, DataLoader
from imutils import paths
import cv2
import numpy as np
import pascal_voc
from pascal_voc import *
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class getDetails(Dataset):
    def __init__(self, img_dir, is_transform=None):
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]

        self.is_transform = is_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        img = cv2.imread(img_name)
 
        iname = img_name.rsplit('\\', 1)[-1].rsplit('.', 1)[0].split('-')
        #print("iname: ", iname)
        [leftUp, rightDown] = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]
        #print("leftUp, rightDown: ", [leftUp, rightDown])
        #print("leftUp: ", leftUp)
        #print("rightDown: ", rightDown)
        ori_w, ori_h = float(img.shape[1]), float(img.shape[0])
        new_labels = [(leftUp[0] + rightDown[0]) / (2 * ori_w), (leftUp[1] + rightDown[1]) / (2 * ori_h),
                      (rightDown[0] - leftUp[0]) / ori_w, (rightDown[1] - leftUp[1]) / ori_h]

        return [leftUp, rightDown], new_labels, img_name

if __name__ == '__main__':
    images = './ccpd_dataset/ccpd_weather'
    Dir = images.split(',')
    dst = getDetails(Dir)
    xmin =[]
    ymin = []
    xmax = []
    ymax = []
    cx = []
    cy = []
    h = []
    w = []
    dataloader = DataLoader(dst, batch_size=1, shuffle=False, num_workers=1)
    for i, ([leftUp, rightDown], bbratio, ims) in enumerate(dataloader):
            print(i)
            #print("Reading image, ", ims)
            #cv2Img = cv2.imread(ims[0])
            #cv2.imshow("Image", cv2Img)
            #cv2.waitKey(0)
            xmin.append(leftUp[0].item())
            ymin.append(leftUp[1].item())
            xmax.append(rightDown[0].item())
            ymax.append(rightDown[1].item())
            cx.append(bbratio[0].item())
            cy.append(bbratio[1].item())
            h.append(bbratio[2].item())
            w.append(bbratio[3].item())
    #print(xmin)
    #print(ymin)
    #print(xmax)
    #print(ymax)
    #print(cx)
    #print(cy)
    #print(h)
    #print(w)
    xminstd = np.std(xmin)
    yminstd = np.std(ymin)
    xmaxstd = np.std(xmax)
    ymaxstd = np.std(ymax)
    cxstd = np.std(cx)
    cystd = np.std(cy)
    hstd = np.std(h)
    wstd = np.std(w)
    xminavg = sum(xmin) / len(xmin)
    xmaxavg = sum(xmax) / len(xmax)
    yminavg = sum(ymin) / len(ymin)
    ymaxavg = sum(ymax) / len(ymax)
    cxavg = sum(cx) / len(cx)
    cyavg = sum(cy) / len(cy)
    havg = sum(h) / len(h)
    wavg = sum(w) / len(w)
    print("Xmin:- ")
    print("Average: ", xminavg)
    print("Standard Deviation: ", xminstd)
    print("Xmax:- ")
    print("Average: ", xmaxavg)
    print("Standard Deviation: ", xmaxstd)
    print("Ymin:- ")
    print("Average: ", yminavg)
    print("Standard Deviation: ", yminstd)
    print("Ymax:- ")
    print("Average: ", ymaxavg)
    print("Standard Deviation: ", ymaxstd)
    print("Cx:- ")
    print("Average: ", cxavg)
    print("Standard Deviation: ", cxstd)
    print("Cy:- ")
    print("Average: ", cyavg)
    print("Standard Deviation: ", cystd) 
    print("H:- ")
    print("Average: ", havg)
    print("Standard Deviation: ", hstd) 
    print("W:- ")
    print("Average: ", wavg)
    print("Standard Deviation: ", wstd) 


    plt.hist(xmin, 400)
    plt.savefig("ChineseDistribution/xmin.png")
    plt.hist(ymin, 400)
    plt.savefig("ChineseDistribution/ymin.png")
    plt.hist(xmax, 400)
    plt.savefig("ChineseDistribution/xmax.png")
    plt.hist(ymax, 400)
    plt.savefig("ChineseDistribution/ymax.png")
    plt.hist(cx, 400)
    plt.savefig("ChineseDistribution/cx.png")
    plt.hist(cy, 400)
    plt.savefig("ChineseDistribution/cy.png")
    plt.hist(h, 400)
    plt.savefig("ChineseDistribution/h.png")
    plt.hist(w, 400)
    plt.savefig("ChineseDistribution/w.png")
    
