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
    def __init__(self, img_dir,  annotationsFolder, is_transform=None):
        print("Enterining dataloader*************")
        self.trainAnnotationsFolder = annotationsFolder
        self.img_dir = img_dir
        #print("img_dir: ", img_dir)
        #print(img_dir)
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
        self.is_transform = is_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        #print("img_name: ", img_name)
        img = cv2.imread(img_name)
# img_name such as 025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
        img_name_without_extension = img_name.split('\\')[-1].split('.', 1)[0]
        #print("Image name: ", img_name_without_extension)
        #lbl = img_name.split('/')[-1].rsplit('.', 1)[0].split('-')[-3]  # Corresponding to the license plate number (0_0_22_27_27_33_16)
        xml_path = self.trainAnnotationsFolder+"/"+ img_name_without_extension+".xml"

        licensePlateText, fileName, boundingBox = readVOC(xml_path)
           
        #print("labelsIndices: ", labelsIndices)

        [leftUp, rightDown] = [[boundingBox[0][0],boundingBox[0][1]],[boundingBox[0][2],boundingBox[0][3]]]  # Corresponding to the coordinates of the upper left corner and the lower right corner
        ori_w, ori_h = [float(int(el)) for el in [img.shape[1], img.shape[0]]]
        new_labels = [(leftUp[0] + rightDown[0]) / (2 * ori_w), (leftUp[1] + rightDown[1]) / (2 * ori_h),
                      (rightDown[0] - leftUp[0]) / ori_w, (rightDown[1] - leftUp[1]) / ori_h]  # Real value [cen_x,cen_y,w,h]
       
        return [leftUp, rightDown], new_labels, img_name

if __name__ == '__main__':
    images = './proper_dataset/v3andv4/padded/image'
    annotations = './proper_dataset/v3andv4/padded/annotation'
    Dir = images.split(',')
    dst = getDetails(Dir, annotations)
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
    plt.savefig("IndianDistribution/xmin.png")
    plt.hist(ymin, 400)
    plt.savefig("IndianDistribution/ymin.png")
    plt.hist(xmax, 400)
    plt.savefig("IndianDistribution/xmax.png")
    plt.hist(ymax, 400)
    plt.savefig("IndianDistribution/ymax.png")
    plt.hist(cx, 400)
    plt.savefig("IndianDistribution/cx.png")
    plt.hist(cy, 400)
    plt.savefig("IndianDistribution/cy.png")
    plt.hist(h, 400)
    plt.savefig("IndianDistribution/h.png")
    plt.hist(w, 400)
    plt.savefig("IndianDistribution/w.png")
