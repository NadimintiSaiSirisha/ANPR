from torch.utils.data import BatchSampler,Dataset, DataLoader
from imutils import paths
import cv2
import numpy as np
import pascal_voc
from pascal_voc import *

# pytorch custom training set data loader
# For the training data
class labelFpsDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, annotationsFolder, is_transform=None):
        print("Enterining dataloader*************")
        self.trainAnnotationsFolder = annotationsFolder
        self.img_dir = img_dir
        #print("img_dir: ", img_dir)
        #print(img_dir)
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
        #print("img_paths: ", self.img_paths)
        # self.img_paths = os.listdir(img_dir)
        # print self.img_paths
        self.img_size = imgSize
        self.is_transform = is_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        print("img_name: ", img_name)
        img = cv2.imread(img_name)
        resizedImage = cv2.resize(img, self.img_size)
        #cv2.imshow("resized "+img_name, resizedImage)
        #cv2.waitKey(0)
        resizedImage = np.transpose(resizedImage, (2, 0, 1))  # [H,W,C]--->[C,H,W]
        resizedImage = resizedImage.astype('float32')
        resizedImage /= 255.0

# img_name such as 025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
        img_name_without_extension = img_name.split('\\')[-1].rsplit('.', 1)[0]
        #print("Image name: ", img_name_without_extension)
        #lbl = img_name.split('/')[-1].rsplit('.', 1)[0].split('-')[-3]  # Corresponding to the license plate number (0_0_22_27_27_33_16)
        xml_path = self.trainAnnotationsFolder+"/"+ img_name_without_extension+".xml"

        licensePlateText, fileName, boundingBox = readVOC(xml_path)
        print("licensePlateText: ", licensePlateText)
        print("fileName: ", fileName)
        
        #print(boundingBox)
        # Creating dictionary to return the correct array indices:
        #    states = ["AN", "AP", "AR", "AS", "BH", "BR", "CH", "PB", "CG", "DD", "DL", "GA", "GJ", "HR", "HP", "JK", "JH", "KA", "KL", "LA",
                    # "LD", "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PY", "PB", "RJ", "SK", "TN", "TS", "TR", "UP", "UK", "WB"]

        dict_state_indices =  {"AN" : 0, "AP": 1, "AR":2, "AS":3, "BH":4, "BR":5, "CH":6, "PB":7, "CG":8, "DD":9, "DL":10, "GA":11, "GJ":12, "HR":13, 
        "HP": 14, "JK":15, "JH":16, "KA":17, "KL":18, "LA":19, "LD":20, "MP":21, "MH":22, "MN":23, "ML":24, "MZ":25, "NL":26, "OD":27, "PY":28, "PB":29, 
        "RJ":30, "SK":31, "TN":32, "TS":33, "TR":34, "UP":35, "UK":36, "WB":37}

        

        dict_digit_indices = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}


        dict_alpha_indices =  {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'J':8, 'K':9, 'L':10, 'M':11, 'N':12, 'P':13, 'Q':14, 'R':15, 'S':16, 'T':17, 'U':18, 'V':19,
                                    'W':20, 'X':21, 'Y':22, 'Z':23}
        labelsIndices=""

        
        state = licensePlateText[0:2]    
        print("State: ", state)
        stateIndex = dict_state_indices[state]
        #print(stateIndex)
        labelsIndices+=str(stateIndex)
        
        digitsText = licensePlateText[2:4]
        print("digitsText: ", digitsText)
        for char in digitsText:
            labelsIndices+="_"+str(dict_digit_indices[char])
        print("Label: ", labelsIndices)
        alphaText = licensePlateText[4:6]
        print("AlphaText: ", alphaText)
        for char in alphaText:
            labelsIndices+="_"+str(dict_alpha_indices[char])
        print("Label: ", labelsIndices)
        digitsText = licensePlateText[6:]
        print("digitsText: ", digitsText)
        for char in digitsText:
            labelsIndices+="_"+str(dict_digit_indices[char])
        print("Label: ", labelsIndices)

        print(licensePlateText)
        print(labelsIndices)
        
        #print("labelsIndices: ", labelsIndices)

        [leftUp, rightDown] = [[boundingBox[0][0],boundingBox[0][1]],[boundingBox[0][2],boundingBox[0][3]]]  # Corresponding to the coordinates of the upper left corner and the lower right corner
        ori_w, ori_h = [float(int(el)) for el in [img.shape[1], img.shape[0]]]
        new_labels = [(leftUp[0] + rightDown[0]) / (2 * ori_w), (leftUp[1] + rightDown[1]) / (2 * ori_h),
                      (rightDown[0] - leftUp[0]) / ori_w, (rightDown[1] - leftUp[1]) / ori_h]  # Real value [cen_x,cen_y,w,h]
        #print("Label indices: ", labelsIndices)
        
        
        return resizedImage, new_labels, labelsIndices, img_name
     

# pytorch custom test set data loader
# For testing the data in train_rpnet.py
class labelTestDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, testAnnotationsPath, is_transform=None):
        self.testAnnotationsFolder = testAnnotationsPath
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
        # self.img_paths = os.listdir(img_dir)
        # print self.img_paths
        self.img_size = imgSize
        self.is_transform = is_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        
        img_name = self.img_paths[index]
        print("Reading image...", img_name)
        img = cv2.imread(img_name)
        resizedImage = cv2.resize(img, self.img_size)
        resizedImage = np.transpose(resizedImage, (2, 0, 1))  # [H,W,C]--->[C,H,W]
        resizedImage = resizedImage.astype('float32')
        resizedImage /= 255.0




# img_name such as 025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
        img_name_without_extension = img_name.split('\\')[-1].rsplit('.', 1)[0]
        #print("Image name: ", img_name_without_extension)
        #lbl = img_name.split('/')[-1].rsplit('.', 1)[0].split('-')[-3]  # Corresponding to the license plate number (0_0_22_27_27_33_16)
        xml_path = self.testAnnotationsFolder+"/"+ img_name_without_extension+".xml"

        licensePlateText, fileName, boundingBox = readVOC(xml_path)
        #print(boundingBox)
        # Creating dictionary to return the correct array indices:
        #    states = ["AN", "AP", "AR", "AS", "BH", "BR", "CH", "PB", "CG", "DD", "DL", "GA", "GJ", "HR", "HP", "JK", "JH", "KA", "KL", "LA",
                    # "LD", "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PY", "PB", "RJ", "SK", "TN", "TS", "TR", "UP", "UK", "WB"]

        dict_state_indices =  {"AN" : 0, "AP": 1, "AR":2, "AS":3, "BH":4, "BR":5, "CH":6, "PB":7, "CG":8, "DD":9, "DL":10, "GA":11, "GJ":12, "HR":13, 
        "HP": 14, "JK":15, "JH":16, "KA":17, "KL":18, "LA":19, "LD":20, "MP":21, "MH":22, "MN":23, "ML":24, "MZ":25, "NL":26, "OD":27, "PY":28, "PB":29, 
        "RJ":30, "SK":31, "TN":32, "TS":33, "TR":34, "UP":35, "UK":36, "WB":37}

        

        dict_digit_indices = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}


        dict_alpha_indices =  {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'J':8, 'K':9, 'L':10, 'M':11, 'N':12, 'P':13, 'Q':14, 'R':15, 'S':16, 'T':17, 'U':18, 'V':19,
                                    'W':20, 'X':21, 'Y':22, 'Z':23}
        labelsIndices=""

        
        state = licensePlateText[0:2]    
        print("State: ", state)
        stateIndex = dict_state_indices[state]
        #print(stateIndex)
        labelsIndices+=str(stateIndex)
        
        digitsText = licensePlateText[2:4]
        print("digitsText: ", digitsText)
        for char in digitsText:
            labelsIndices+="_"+str(dict_digit_indices[char])
        print("Label: ", labelsIndices)
        alphaText = licensePlateText[4:6]
        print("AlphaText: ", alphaText)
        for char in alphaText:
            labelsIndices+="_"+str(dict_alpha_indices[char])
        print("Label: ", labelsIndices)
        digitsText = licensePlateText[6:]
        print("digitsText: ", digitsText)
        for char in digitsText:
            labelsIndices+="_"+str(dict_digit_indices[char])
        print("Label: ", labelsIndices)

        print(licensePlateText)
        print(labelsIndices)
        

        return resizedImage, labelsIndices, img_name

# For loading the data in wR2.py
class ChaLocDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, annotationsFolder, is_transform=None):
        print("Enterining dataloader*************")
        self.trainAnnotationsFolder = annotationsFolder
        self.img_dir = img_dir
        #print("img_dir: ", img_dir)
        #print(img_dir)
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
        #print("img_paths: ", self.img_paths)
        # self.img_paths = os.listdir(img_dir)
        # print self.img_paths
        self.img_size = imgSize
        self.is_transform = is_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        #print("img_name: ", img_name)
        img = cv2.imread(img_name)
        resizedImage = cv2.resize(img, self.img_size)
        #cv2.imshow("resized "+img_name, resizedImage)
        #cv2.waitKey(0)
        resizedImage = np.transpose(resizedImage, (2, 0, 1))  # [H,W,C]--->[C,H,W]
        resizedImage = resizedImage.astype('float32')
        resizedImage /= 255.0

# img_name such as 025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
        img_name_without_extension = img_name.split('\\')[-1].rsplit('.', 1)[0]
        #print("Image name: ", img_name_without_extension)
        #lbl = img_name.split('/')[-1].rsplit('.', 1)[0].split('-')[-3]  # Corresponding to the license plate number (0_0_22_27_27_33_16)
        xml_path = self.trainAnnotationsFolder+"/"+ img_name_without_extension+".xml"

        licensePlateText, fileName, boundingBox = readVOC(xml_path)
           
        #print("labelsIndices: ", labelsIndices)

        [leftUp, rightDown] = [[boundingBox[0][0],boundingBox[0][1]],[boundingBox[0][2],boundingBox[0][3]]]  # Corresponding to the coordinates of the upper left corner and the lower right corner
        ori_w, ori_h = [float(int(el)) for el in [img.shape[1], img.shape[0]]]
        new_labels = [(leftUp[0] + rightDown[0]) / (2 * ori_w), (leftUp[1] + rightDown[1]) / (2 * ori_h),
                      (rightDown[0] - leftUp[0]) / ori_w, (rightDown[1] - leftUp[1]) / ori_h]  # Real value [cen_x,cen_y,w,h]
       
        return resizedImage, new_labels, img_name

# For loading the testing data in the demo (only contains the image and the image name)
class demoTestDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, is_transform=None):
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
        self.img_size = imgSize
        self.is_transform = is_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        img = cv2.imread(img_name)
        # img = img.astype('float32')
        resizedImage = cv2.resize(img, self.img_size)
        resizedImage = np.transpose(resizedImage, (2, 0, 1))
        resizedImage = resizedImage.astype('float32')
        resizedImage /= 255.0
        return resizedImage, img_name