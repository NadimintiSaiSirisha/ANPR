from torch.utils.data import BatchSampler,Dataset, DataLoader
from imutils import paths
import cv2
import numpy as np
import pascal_voc


# pytorch custom training set data loader
class labelFpsDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, is_transform=None):
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
        img = cv2.imread(img_name)
        resizedImage = cv2.resize(img, self.img_size)
        resizedImage = np.transpose(resizedImage, (2, 0, 1))  # [H,W,C]--->[C,H,W]
        resizedImage = resizedImage.astype('float32')
        resizedImage /= 255.0    
        # img_name such as 025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
        lbl = img_name.split('/')[-1].rsplit('.', 1)[0].split('-')[-3]  # Corresponding license plate number (0_0_22_27_27_33_16)
        print("lbl: ", lbl)
        iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
        print("iname: ", iname)
        [leftUp, rightDown] = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]  # Corresponding to the coordinates of the upper left and lower right corners
        print("[leftUp, rightDown]: ", [leftUp, rightDown])
        ori_w, ori_h = [float(int(el)) for el in [img.shape[1], img.shape[0]]]
        print("ori_w, ori_h: ", ori_w, ori_h)
        new_labels = [(leftUp[0] + rightDown[0]) / (2 * ori_w), (leftUp[1] + rightDown[1]) / (2 * ori_h),
                      (rightDown[0] - leftUp[0]) / ori_w, (rightDown[1] - leftUp[1]) / ori_h]  # True value [cen_x,cen_y,w,h]
        print("[leftUp, rightDown] = ", [leftUp, rightDown])
        # [leftUp, rightDown] =  [[277, 540], [469, 604]]
        print("new_labels:", new_labels)
        # new_labels: [0.5180555555555556, 0.49310344827586206, 0.26666666666666666, 0.05517241379310345]
        print("type(new_labels) :", type(new_labels))
        # type(new_labels) : <class 'list'>
        print("lbl: ", lbl)
        # lbl:  15_0_20_33_19_26_24
        print("type(lbl): ", type(lbl))
        # type(lbl):  <class 'str'>
        
        return resizedImage, new_labels, lbl, img_name

# pytorch custom test set data loader
class labelTestDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, is_transform=None):
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
        img = cv2.imread(img_name)
        # img = img.astype('float32')
        resizedImage = cv2.resize(img, self.img_size)
        resizedImage = np.transpose(resizedImage, (2, 0, 1))
        resizedImage = resizedImage.astype('float32')
        resizedImage /= 255.0
        lbl = img_name.split('/')[-1].split('.')[0].split('-')[-3]
        return resizedImage, lbl, img_name

# The functions len and getitem are called as per the dataloader object
class ChaLocDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, is_transform=None):
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
        img = cv2.imread(img_name)
        resizedImage = cv2.resize(img, self.img_size)
        # image is in the form of a 3d tensor
        resizedImage = np.reshape(resizedImage, (resizedImage.shape[2], resizedImage.shape[0], resizedImage.shape[1]))

        iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
        print("iname: ", iname)
        [leftUp, rightDown] = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]
        print("leftUp, rightDown: ", [leftUp, rightDown])
        #print("leftUp: ", leftUp)
        #print("rightDown: ", rightDown)
        ori_w, ori_h = float(img.shape[1]), float(img.shape[0])
        assert img.shape[0] == 1160
        new_labels = [(leftUp[0] + rightDown[0]) / (2 * ori_w), (leftUp[1] + rightDown[1]) / (2 * ori_h),
                      (rightDown[0] - leftUp[0]) / ori_w, (rightDown[1] - leftUp[1]) / ori_h]

        resizedImage = resizedImage.astype('float32')
        resizedImage /= 255.0
        return resizedImage, new_labels, img_name


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
