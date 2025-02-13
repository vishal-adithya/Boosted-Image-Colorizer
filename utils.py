# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 23:42:30 2025

@author: vishaladithyaa
"""

import cv2
import numpy as np


def Feature_Extractor(grayscaled_img):
    
    height,width = grayscaled_img.shape
    x_grid,y_grid = np.meshgrid(np.linspace(0, 1,width),np.linspace(0, 1,height))
    
    grayscaled_img = grayscaled_img.flatten()
    x_grid = x_grid.flatten()
    y_grid = y_grid.flatten()
    
    feature_stack = np.column_stack((grayscaled_img,x_grid,y_grid))
    return feature_stack

def Preprocessing(img_path,img_size):
    
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)
    
    grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img,grayscaled