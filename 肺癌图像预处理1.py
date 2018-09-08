from __future__ import print_function, division
import os
import SimpleITK as sitk
import numpy as np
from glob import glob
import pandas as pd
import cv2
import random
import scipy.ndimage
try:
    from tqdm import tqdm  # long waits are not fun
except:
    print('TQDM does make much nicer wait bars...')
    tqdm = lambda x: x

subset = "train_subset*/"
#subset = "subset*/"###while need to preprocess luna ,chose this
#subset = "val_subset*/"

tianchi_path = "./tianchi"
#tianchi_path = "./LUNA 2016"###while need to preprocess luna16 ,chose this


output_path = "./tianchi/tianchi-2D/"

def make_mask(center, diam, z, width, height, spacing, origin):
    '''
Center : centers of circles px -- list of coordinates x,y,z
diam : diameters of circles px -- diameter
widthXheight : pixel dim of image
spacing = mm/px conversion rate np array x,y,z
origin = x,y,z mm np.array
z = z position of slice in world coordinates mm
    '''
    mask = np.zeros([height, width])  # 0's everywhere except nodule swapping x,y to match img
    # convert to nodule space from world coordinates

    # Defining the voxel range in which the nodule falls
    v_center = (center - origin) / spacing
    v_diam = int(diam / spacing[0])
    v_xmin = np.max([0, int(v_center[0] - v_diam)])
    v_xmax = np.min([width - 1, int(v_center[0] + v_diam)])
    v_ymin = np.max([0, int(v_center[1] - v_diam) ])
    v_ymax = np.min([height - 1, int(v_center[1] + v_diam) ])

    v_xrange = range(v_xmin, v_xmax + 1)
    v_yrange = range(v_ymin, v_ymax + 1)

    # Convert back to world coordinates for distance calculation
    x_data = [x * spacing[0] + origin[0] for x in range(width)]
    y_data = [x * spacing[1] + origin[1] for x in range(height)]

    # Fill in 1 within sphere around nodule
    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0] * v_x + origin[0]
            p_y = spacing[1] * v_y + origin[1]
            if np.linalg.norm(center - np.array([p_x, p_y, z])) <= diam:
                mask[int((p_y - origin[1]) / spacing[1]), int((p_x - origin[0]) / spacing[0])] = 1.0
    return (mask)

def matrix2int16(matrix):
    ''' 
matrix must be a numpy array NXN
Returns uint16 version
    '''
    m_min = np.min(matrix)
    m_max = np.max(matrix)
    matrix = matrix - m_min
    return (np.array(np.rint((matrix - m_min) / float(m_max - m_min) * 65535.0), dtype=np.uint16))

train_data_path = os.path.join(tianchi_path, subset)

#print("train_data_path: %s" % train_data_path)
train_images = glob(train_data_path + "*.mhd")
#print(train_images)

tmp_workspace = os.path.join(output_path, "traindata/")
tmp_label_workspace = os.path.join(output_path, "trainlabel/")
if not os.path.exists(tmp_workspace):
    os.makedirs(tmp_workspace)
if not os.path.exists(tmp_label_workspace):
    os.makedirs(tmp_label_workspace)

#
# Helper function to get rows in data frame associated
# with each file
def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return (f)


#
# The locations of the nodes
df_node = pd.read_csv(tianchi_path + "/csv/val/annotations.csv")
# df_node = pd.read_csv(tianchi_path + "/CSVFILES/annotations.csv")
df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(train_images, file_name))
df_node = df_node.dropna()

for fcount, img_file in enumerate(tqdm(train_images)):
    mini_df = df_node[df_node["file"] == img_file]  # get all nodules associate with file
    if True:#mini_df.shape[0] > 0:
        itk_img = sitk.ReadImage(img_file)
        img_array = sitk.GetArrayFromImage(itk_img)  # indexes are z,y,x (notice the ordering)
        num_z, height, width = img_array.shape  # heightXwidth constitute the transverse plane
#         if width>512:
#             continue
        origin = np.array(itk_img.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
        spacing = np.array(itk_img.GetSpacing())  # spacing of voxels in world coor. (mm)
        
        masks=np.zeros([num_z, height, width], dtype=np.uint8)
        
        maskdir=[]
        
        for node_idx, cur_row in mini_df.iterrows():
            node_x = cur_row["coordX"]
            node_y = cur_row["coordY"]
            node_z = cur_row["coordZ"]
            diam = cur_row["diameter_mm"]
            
            
            center = np.array([node_x, node_y, node_z])
            v_center = np.rint((center - origin) / spacing).astype(np.int)
            
            for i, i_z in enumerate(np.arange(int(v_center[2] - max(1,0.5*diam/spacing[2])),
                                                  int(v_center[2] +1+ max(1,0.5*diam/spacing[2]))).clip(0,
                                                                             num_z - 1)):
                mask = make_mask(center, max(0,(diam**2-(2*(v_center[2]-i_z)*spacing[2])**2)**0.5), i_z * spacing[2] + origin[2],
                                 width, height, spacing, origin)
                masks[i_z] = masks[i_z]+mask
        if mini_df.shape[0]>0:
            masks[masks>1]=1
            masks[img_array<-700]=0
            masks=masks.astype(np.uint8)
        img_array[img_array<-1024]=-1024
        img_array[img_array>1024]=1024
        img_array=img_array/1000+0.5
        img_array=img_array.astype(np.float32)
        spacing=np.array([spacing[2],spacing[1],spacing[0]])
        img_array=scipy.ndimage.interpolation.zoom(img_array, spacing)
        masks=scipy.ndimage.interpolation.zoom(masks, spacing)
        img_array=np.transpose(img_array,[1,2,0])
        ox,oy,oz=origin
        masks=np.transpose([masks,1-masks],[2,3,1,0])
        name=img_file.split('\\')[-1][:-4]+'_%.2f_%.2f_%.2f_.npy'%(oy,ox,oz)
        np.save(tmp_workspace+name,img_array)
        np.save(tmp_label_workspace+name,masks)
        
