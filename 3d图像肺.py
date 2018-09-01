# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 15:18:29 2018

@author: hasee
"""
import cv2
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.pyplot as plt  
import numpy as np  
import pydicom
z=np.array([],int)
y=np.array([],int)
x=np.array([],int)
postx=0
posty=0
before="D:/CT图像/2-丁20150806000874/17106B6C134F4C47B7607C92EBD5862E/ ("
after=").dcm"
for zz in range(1,155):
    dcm = pydicom.read_file(before+str(zz)+after)
    dcm.image = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    slices = []
    slices.append(dcm)
    i=500
    img = slices[ int(len(slices)/2) ].image.copy()
    img[(img>=-i)]=255
    img[(img<-i)]=0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    img2= slices[ int(len(slices)/2) ].image.copy()
    ret,img2= cv2.threshold(img2,-95,70, cv2.THRESH_BINARY)
    img2= np.uint8(img2)
    im2, contours, _ = cv2.findContours(img2,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(img.shape, np.uint8)
    for contour in contours:
        cv2.fillPoly(mask, [contour], 255)
    img2[(mask > 0)] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    img2= cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel)

    img2[(img==255)]=0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    img2= cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel)
    img2= cv2.Canny(img2,30,150)
    xs=np.array([],int)
    ys=np.array([],int)
    for eachline in img2:
        xss=np.array([],int)
        for eachone in eachline:
            if (eachone==255):
                xss=np.append(xss,postx)
            yss=np.zeros(len(xss))+posty
            postx+=1
        xs=np.append(xs,xss)
        ys=np.append(ys,yss)
        postx=0
        posty+=1
    zs=np.zeros(len(xs))+zz
    posty=0
    z=np.append(z,zs)
    x=np.append(x,xs)
    y=np.append(y,ys)
fig=plt.figure(dpi=120)  
ax=fig.add_subplot(111,projection='3d')  
plt.title('point cloud')  
ax.scatter(x,y,z,c='b',marker='.',s=2,linewidth=0,alpha=1,cmap='spectral') 
ax.axis('scaled')
plt.show()
