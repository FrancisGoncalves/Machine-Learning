import cv2
import glob 
import os
from pathlib import Path

path = '.'
# HOME_PATH = glob.glob(path + '/*.JPG')
DESTINATION_PATH = "C:\\Users\\Francisco Gonçalves\\Desktop"
# print(HOME_PATH)
i = 3
# for img in HOME_PATH:
os.chdir("C:\\Users\\Francisco Gonçalves\\Desktop")
# print(img)
img = cv2.imread('IMG_3949.JPG')
new_image = cv2.resize(img, (416, 416))
filename = "white_boat" + str(i) + "_resized.jpg"
i += 1
os.chdir(DESTINATION_PATH)
cv2.imwrite(filename, new_image)
print(i)