from torch.utils.data import BatchSampler,Dataset, DataLoader
from imutils import paths
import cv2
import numpy as np
from pascal_voc import readVOC

img_name = "car-wbs-AP02BP2454_00000.jpg"
#img = cv2.imread(img_name)
#img_size = (480,480)
#resizedImage = cv2.resize(img, img_size)
#resizedImage = np.transpose(resizedImage, (2, 0, 1))  # [H,W,C]--->[C,H,W]
#resizedImage = resizedImage.astype('float32')
#resizedImage /= 255.0

trainAnnotations = './data_indian/train/annotations'


# img_name such as 025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
img_name_without_extension = img_name.split('/')[-1].rsplit('.', 1)[0]
print("Image name: ", img_name_without_extension)
#lbl = img_name.split('/')[-1].rsplit('.', 1)[0].split('-')[-3]  # Corresponding to the license plate number (0_0_22_27_27_33_16)
xml_path = trainAnnotations+"/"+ img_name_without_extension+".xml"

licensePlateText, fileName, boundingBox = readVOC(xml_path)
print(boundingBox)
# Creating dictionary to return the correct array indices:
#    states = ["AN", "AP", "AR", "AS", "BH", "BR", "CH", "PB", "CG", "DD", "DL", "GA", "GJ", "HR", "HP", "JK", "JH", "KA", "KL", "LA",
            # "LD", "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PY", "PB", "RJ", "SK", "TN", "TS", "TR", "UP", "UK", "WB"]

dict_state_indices =  {"AN" : 0, "AP": 1, "AR":2, "AS":3, "BH":4, "BR":5, "CH":6, "PB":7, "CG":8, "DD":9, "DL":10, "GA":11, "GJ":12, "HR":13, 
"HP": 14, "JK":15, "JH":16, "KA":17, "KL":18, "LA":19, "LD":20, "MP":21, "MH":22, "MN":23, "ML":24, "MZ":25, "NL":26, "OD":27, "PY":28, "PB":29, 
"RJ":30, "SK":31, "TN":32, "TS":33, "TR":34, "UP":35, "UK":36, "WB":37}

dict_alphaNum_indices =  {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'J':8, 'K':9, 'L':10, 'M':11, 'N':12, 'P':13, 'Q':14, 'R':15, 'S':16, 'T':17, 'U':18, 'V':19,
                             'W':20, 'X':21, 'Y':22, 'Z':23, '0':24, '1':25, '2':26, '3':27, '4':28, '5':29, '6':30, '7':31, '8':32, '9':33, 'O':34, ' ':35}
labelsIndices = []

state = licensePlateText[0:2]    
stateIndex = dict_state_indices[state]
print(stateIndex)
labelsIndices.append(stateIndex)

substringLP = licensePlateText[2:]
for char in substringLP:
    labelsIndices.append(dict_alphaNum_indices[char])

print(labelsIndices)

[leftUp, rightDown] = [[boundingBox[0][0],boundingBox[0][1]],[boundingBox[0][2],boundingBox[0][3]]]  # Corresponding to the coordinates of the upper left corner and the lower right corner
print([leftUp, rightDown])
#ori_w, ori_h = [float(int(el)) for el in [img.shape[1], img.shape[0]]]
#new_labels = [(leftUp[0] + rightDown[0]) / (2 * ori_w), (leftUp[1] + rightDown[1]) / (2 * ori_h),
 #               (rightDown[0] - leftUp[0]) / ori_w, (rightDown[1] - leftUp[1]) / ori_h]  # Real value [cen_x,cen_y,w,h]