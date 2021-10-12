import cv2
import numpy as np
from imutils import paths
from pascal_voc import *

def letterbox_image(image, size, boundingbox):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.shape[0:2][::-1]
    print("iw, ih: ", iw,ih)
    w, h = size
    scale = min(w/iw, h/ih)
    print("scale: ",scale)
    nw = int(iw*scale)
    print("nw: ", nw)
    nh = int(ih*scale)
    print("nh: ", nh)
    image = cv2.resize(image, (nw,nh), interpolation=cv2.INTER_CUBIC)
    print("size[1]: ", size[1]) # size[1] is the height of the image: 1160
    print("size[0]: ", size[0]) # size[0] is the width of the image: 720
    new_image = np.zeros((size[1], size[0], 3), np.uint8)
    new_image.fill(0)
    dx = (w-nw)//2
    dy = (h-nh)//2
    new_image[dy:dy+nh, dx:dx+nw,:] = image

    #Calculating new bounding box
    resized_bb = [element * scale for element in boundingbox[0]]
    print(resized_bb)
    resized_bb[0]+=dx
    resized_bb[1]+=dy
    resized_bb[2]+=dx
    resized_bb[3]+=dy
    

    return new_image, resized_bb

images = './proper_dataset/v3andv4/testAnalysis'
size = (720,1160)
annotationsFolder = './proper_dataset/olx/combined/annotation'
dstFolder = './proper_dataset/olx/combined/padded'
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
    lp, filename, list_with_all_boxes = readVOC(annotationsFolder+'/'+img_name_without_extension+".xml")
    print(lp)
    print(filename)
    print(list_with_all_boxes)
    
    image = cv2.imread(img_name)
    resized_image, resized_bb=letterbox_image(image, size, list_with_all_boxes)
    
    #cv2.rectangle(resized_image, (int(resized_bb[0]), int(resized_bb[1])), (int(resized_bb[2]), int(resized_bb[3])), (0, 255, 0),2)
    #cv2.imshow('Padded image', resized_image)
    cv2.waitKey(0)
    cv2.imwrite(dstFolder+'/'+img_name_only,resized_image)
    createVOC(dstFolder, img_name_only, str(resized_image.shape[1]), str(resized_image.shape[0]), str(resized_image.shape[2]), lp, str(int(resized_bb[0])), str(int(resized_bb[1])), str(int(resized_bb[2])), str(int(resized_bb[3])) )  
    
    #cv2.waitKey(0)