import numpy as np
import os
import cv2
from random import choice
from PIL import Image
import matplotlib.pyplot as plt

img = cv2.imread('test.jpeg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
name = 'Khuong Duy'
dir = 'Data/train/' + name + '/'
filename = choice([filename for filename in os.listdir(dir)])
path = dir + filename
img1 = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
plt.subplot(1,2,1)
plt.axis('off')
plt.imshow(img)
plt.title('Ten')
plt.subplot(1,2,2)
plt.axis('off')
plt.imshow(img1)
plt.title('Anh so sanh')
plt.savefig('static/rs.jpg')
