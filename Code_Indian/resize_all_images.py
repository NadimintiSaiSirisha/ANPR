import cv2

from imutils import paths

images = './v4/images'
img_dir = images.split(',')
img_paths=[]
for i in range(len(img_dir)):
    img_paths += [el for el in paths.list_images(img_dir[i])]
print(img_paths)

dstFolder = './v4_resized/'
for index in range(len(img_paths)):
    img_name = img_paths[index]
    print(img_name)
    img = cv2.imread(img_name)
    img_size = (720,1160)
    resizedImage = cv2.resize(img, img_size)
    img_name_only = img_name.split('\\')[-1]
    print(img_name_only)
    #cv2.imshow("resized", resizedImage)
    cv2.imwrite(dstFolder+img_name_only,resizedImage)
    #cv2.waitKey(0)

    