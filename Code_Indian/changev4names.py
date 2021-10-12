from imutils import paths
import os
import shutil

path = './v3andv4/annotation'
dstpath = './v3andv4/proper/'
xml_dir = path.split(',')
xml_paths = []
for i in range(len(xml_dir)):
    for el in paths.list_files(xml_dir[i]):
        print(el)
        file_name = el.rsplit('\\',1)[-1].split('.')[0]
        print("filename: ", file_name)
        shutil.copyfile(el, dstpath+file_name+".xml")
