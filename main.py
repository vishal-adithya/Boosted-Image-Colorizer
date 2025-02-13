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

# feature extraction
height,width = gray.shape
x_grid,y_grid = np.meshgrid(np.linspace(0,1,width),np.linspace(0, 1,height))
gray_flat = gray.flatten()
x_grid_flat = x_grid.flatten()
y_grid_flat = y_grid.flatten()

fig,ax = plt.subplots(ncols=2)
ax[0].imshow(x_grid)
ax[1].imshow(y_grid)
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
n = 0
for filename in os.listdir(folder):
    n+=1
    if n>10000:
        break
    path = os.path.join(folder,filename)
    img = cv2.imread(path)
    if img is None:
        continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"[{n}] - image loaded")
    
    feature_stack = Feature_Extraction(gray_img=gray)
    X_data.append(feature_stack)
    print(f"[{n}] - features extracted!!")

    tar_red = img[:,:,2].flatten()
    tar_green = img[:,:,1].flatten() 
    tar_blue = img[:,:,0].flatten()
    
    y_r.append(tar_red)
    y_g.append(tar_green)
    y_b.append(tar_blue)
    print(f"[{n}] - data appended!!")


