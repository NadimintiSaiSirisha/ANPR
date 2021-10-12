import cv2
import numpy as np
from imutils import paths
from pascal_voc import *

# load the image and display it to our screen
images = './proper_dataset/olx/combined/padded/image'
annotationsFolder = './proper_dataset/olx/combined/padded/annotation'
dstFolder = './proper_dataset/olx/combined/shifted/'
img_dir = images.split(',')
img_paths=[]
for i in range(len(img_dir)):
    img_paths += [el for el in paths.list_images(img_dir[i])]
print(img_paths)

for index in range(len(img_paths)):
    img_name = img_paths[index]
    image = cv2.imread(img_name)
    cv2.imshow("Original", image)
    img_name_only = img_name.split('\\')[-1]
    print(img_name_only)
    
    img_name_without_extension = img_name_only.split('.')[0]
    print("img_name_without_extension: ", img_name_without_extension)
    lp, filename, list_with_all_boxes = readVOC(annotationsFolder+'/'+img_name_without_extension+".xml")
    xminActual = list_with_all_boxes[0][0]
    yminActual = list_with_all_boxes[0][1]
    # Translate by the difference of xmin to required xmin and ymin to required ymin
    dx = 263 - xminActual
    print("dx: ", dx)
  
    dy = 478 - yminActual
    print("dy: ", dy)
    #cv2.waitKey(0)
    # shift the image 25 pixels to the right and 50 pixels down
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    #cv2.imshow("Shifted Down and Right", shifted)
    #cv2.waitKey(0)
    print(dstFolder+img_name_only)
    cv2.imwrite(dstFolder+img_name_only, shifted)
    createVOC(dstFolder, img_name_only, str(shifted.shape[1]), str(shifted.shape[0]), str(shifted.shape[2]), lp, str(int(list_with_all_boxes[0][0] + dx)), str(int(list_with_all_boxes[0][1]+dy)), str(int(list_with_all_boxes[0][2]+dx)), str(int(list_with_all_boxes[0][3]+dy)) )