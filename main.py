#dependencies
import os
import cv2
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

# removing unecessary files
folder = "Data/"
for file in os.listdir(folder):
    if file.endswith(".txt"):
        os.remove(os.path.join(folder,file))

img = cv2.imread("ref_img.jpg")
plt.imshow(img)
plt.title("reference image BGR")
plt.show()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img,cmap = "gray")
plt.title("reference image GRAY")
plt.show()

def Preprocessing(img_path,img_size = (256,256)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img,gray


