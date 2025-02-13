import os
import cv2
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from datasets import load_dataset

dataset = load_dataset("rafaelpadilla/coco2017", split="train")


