# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 00:29:46 2021

@author: Siri
"""


#!/usr/bin/env python
# coding:utf-8
 
#from xml.etree.ElementTree import Element, SubElement, tostring
from lxml.etree import Element, SubElement, tostring
import pprint
from xml.dom.minidom import parseString
 
#def createVOC(folder_name, img_name, img_width, img_height, img_depth, lpnumber, xmin, ymin , xmax, ymax):
node_root = Element('annotation') # <annotation>
 
node_folder = SubElement(node_root, 'folder')
node_folder.text = 'images' # 	<folder>images</folder>
 
img_name = 'input_image-0.jpg'
node_filename = SubElement(node_root, 'filename')
node_filename.text = img_name # 	<filename>input_image-0.jpg</filename>
 

node_path = SubElement(node_root, 'path') # 	<path></path>
node_path.text =''

node_source = SubElement(node_root, 'source') # 		<source>
node_database = SubElement(node_source, 'database')  #<database>Unknown</database>
node_database.text = 'Unknown' 
# 	</source>


# =============================================================================
#     	<size>
# 		<width>1920</width>
# 		<height>1080</height>
# 		<depth>3</depth>
# 	</size>
# =============================================================================
node_size = SubElement(node_root, 'size')
node_width = SubElement(node_size, 'width')
node_width.text = '500'
 
node_height = SubElement(node_size, 'height')
node_height.text = '375'
 
node_depth = SubElement(node_size, 'depth')
node_depth.text = '3'
 
# <segmented>0</segmented>

node_segmented = SubElement(node_root, 'segmented')
node_segmented.text = '0' # 


# =============================================================================
#     <object>
# 		<name>MH03AN0055</name>
# 		<pose>Unspecified</pose>
# 		<truncated>0</truncated>
# 		<difficult>0</difficult>
# 		<bndbox>
# 			<xmin>550</xmin>
# 			<ymin>513</ymin>
# 			<xmax>676</xmax>
# 			<ymax>548</ymax>
# 		</bndbox>
# 	</object>
# </annotation>
# 
# 
# =============================================================================
node_object = SubElement(node_root, 'object')

node_name = SubElement(node_object, 'name')
node_name.text = 'MH03AN0055'

node_pose = SubElement(node_object, 'pose')
node_pose.text = 'Unspecified'

node_truncated = SubElement(node_object, 'truncated')
node_truncated.text = '0'


node_difficult = SubElement(node_object, 'difficult')
node_difficult.text = '0'

node_bndbox = SubElement(node_object, 'bndbox')
node_xmin = SubElement(node_bndbox, 'xmin')
node_xmin.text = '99'
node_ymin = SubElement(node_bndbox, 'ymin')
node_ymin.text = '358'
node_xmax = SubElement(node_bndbox, 'xmax')
node_xmax.text = '135'
node_ymax = SubElement(node_bndbox, 'ymax')
node_ymax.text = '375'
 
Xml = tostring(node_root, pretty_print=True) #Formatted display, the newline of the newline
dom = parseString(Xml)
print (Xml)
print(type(Xml))
print(img_name.rsplit( ".", 1 )[ 0 ])
name = img_name.rsplit( ".", 1 )[ 0 ]
print(type(name))
f = open("%s.xml" % name, "wb")
f.write(Xml)
f.close()