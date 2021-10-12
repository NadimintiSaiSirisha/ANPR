import cv2
import numpy as np
from imutils import paths
from pascal_voc import *


images = './proper_dataset/rpnetAll/image'

annotationsFolder = './proper_dataset/rpnetAll/label'
dstFolder = './proper_dataset/rpnet10/image'
img_dir = images.split(',')
img_paths=[]
for i in range(len(img_dir)):
    img_paths += [el for el in paths.list_images(img_dir[i])]
print(img_paths)

for index in range(len(img_paths)):
    img_name = img_paths[index]
    print(img_name)
    img_name_only = img_name.split('\\')[-1]
    print(img_name_only)
    img_name_without_extension = img_name_only.split('.')[0]
    print("img_name_without_extension: ", img_name_without_extension)
    lp, filename, bb = readVOC(annotationsFolder+'/'+img_name_without_extension+".xml")
    list_with_all_boxes = bb[0]
    if(len(lp) == 10):

    
        image = cv2.imread(img_name)
    
    #cv2.rectangle(image, (int(list_with_all_boxes[0]), int(list_with_all_boxes[1])), (int(list_with_all_boxes[2]), int(list_with_all_boxes[3])), (0, 255, 0),2)
    #cv2.imshow('Padded image', image)
    #cv2.waitKey(0)
        cv2.imwrite(dstFolder+'/'+img_name_only,image)
