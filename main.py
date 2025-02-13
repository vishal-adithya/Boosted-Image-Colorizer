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
plt.imshow(gray,cmap = "gray")
plt.title("reference image GRAY")
plt.show()

def Preprocessing(img_path,img_size = (256,256)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img,gray

# feature extraction
height,width = gray.shape
x_grid,y_grid = np.meshgrid(np.linspace(0,1,width),np.linspace(0, 1,height))
gray_flat = gray.flatten()
x_grid_flat = x_grid.flatten()
y_grid_flat = y_grid.flatten()

plt.imshow(x_grid)
plt.title("X Grid")
plt.show()
plt.imshow(y_grid)
plt.title("Y Grid")
plt.show()

def Feature_Extraction(gray_img):
    height,width = gray_img.shape
    x,y = np.meshgrid(np.linspace(0, 1,width),np.linspace(0, 1,height))
    gray_img_flat = gray_img.flatten()
    x_falt = x.flatten()
    y_flat = y.flatten()
    
    feature_stack = np.column_stack((gray_img_flat,x_falt,y_flat))
    return feature_stack

y_r = []
y_g = []
y_b = []
X_data = []

