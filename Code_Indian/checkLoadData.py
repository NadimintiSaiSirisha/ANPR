from torch.utils.data import *
from imutils import paths
import cv2
import numpy as np
img_name ='01-90_89-228&482_412&537-406&529_233&531_232&485_405&483-0_0_21_23_27_29_33-154-14'
img = cv2.imread('./01-90_89-228&482_412&537-406&529_233&531_232&485_405&483-0_0_21_23_27_29_33-154-14.jpg')
cv2.imshow('img',img)
cv2.waitKey(0) 
imgSize = (480,480)
resizedImage = cv2.resize(img, imgSize)
resizedImage = np.reshape(resizedImage, (resizedImage.shape[2], resizedImage.shape[0], resizedImage.shape[1]))

iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
[leftUp, rightDown] = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]


ori_w, ori_h = float(img.shape[1]), float(img.shape[0])
assert img.shape[0] == 1160
new_labels = [(leftUp[0] + rightDown[0]) / (2 * ori_w), (leftUp[1] + rightDown[1]) / (2 * ori_h),
              (rightDown[0] - leftUp[0]) / ori_w, (rightDown[1] - leftUp[1]) / ori_h]

resizedImage = resizedImage.astype('float32')
resizedImage /= 255.0
