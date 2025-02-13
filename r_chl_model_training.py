# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 01:55:05 2025

@author: vishaladithyaa
"""

import os
import cv2
import numpy as np
from utils import *
import xgboost as xgb
import matplotlib.pyplot as plt


img = cv2.imread("ref.jpg")
plt.imshow(img)
plt.title("reference image BGR")
plt.show()

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h,w = img.shape

x,y = np.meshgrid(np.linspace(0, 1,w),np.linspace(0, 1,h))

plt.imshow(x)
plt.title("Meshgrid: X-axis")
plt.show()
plt.imshow(y)
plt.title("Meshgrid: Y-axis")
plt.show()

X = []
y = []
folder = os.path.join("Data","images","Train")
n = 0
for filename in os.listdir(folder):
    filepath = os.path.join(folder,filename)
    img,gray = Preprocessing(filepath, (256,256))
    n+=1
    if img is None:
        continue
    feature_stack = Feature_Extractor(gray)
    X.append(feature_stack)
    y.append(img[:,:,2].flatten())
    print(f"[IMAGE: {n}] ===================================== [DONE!]")
    
X = np.vstack(X)
y = np.hstack(y)
dmatrix = xgb.DMatrix(X,label=y)

params = {"objective":"reg:squarederror","n_estimators" : 200,"booster":"gblinear"}
params["tree_method"] = "grow_gpu_hist"
model = xgb.train(params = params,num_boost_round=5,dtrain = dmatrix,verbose_eval=1)
