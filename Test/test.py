from mtcnn import MTCNN
import numpy as np
import os
from PIL import Image
import cv2
from keras.models import load_model
import pickle
import imutils
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer, LabelEncoder
import time

# Load anh
image = Image.open('test.jpg')
img = cv2.imread('test.jpg')
# Chuyen sang anh mau neu can
image = image.convert('RGB')
# Chuyen du lieu thanh mang
pixels = np.asarray(image)
# Tao bo detector
detector = MTCNN()
# Load FaceNet model
model = load_model('facenet_keras.h5')
# Load model da duoc train
clf = pickle.loads(open('output/clf.pkl', "rb").read())
lb = pickle.loads(open('output/lb.pkl', "rb").read())
# Bo chuan hoa
def normalization(vector):
    a = 0
    x = []
    for i in range(len(vector)):
        a = np.sqrt(a**2 + vector[i]**2)
    for i in range(len(vector)):
        vector[i] = vector[i]/a
    return vector

start = time.time()
# Phat hien khuon mat trong buc anh
results = detector.detect_faces(pixels)
while True:
    for i in range(len(results)):
        x1, y1, width, weight = results[i]['box']
        # Cat khuon mat
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + weight
        face = pixels[y1:y2, x1:x2]
        face = Image.fromarray(face)
        face = face.resize((160,160))
        face = np.asarray(face).astype('float32')
        # Chuan hoa cac gia tri pixels
        mean, std = face.mean(), face.std()
        face_pixels = (face - mean) / std
        sample = np.expand_dims(face_pixels, axis=0)
        embeddingFace = model.predict(sample)[0]
        face = np.asarray(embeddingFace)
        face = normalization(face)
        sample = np.expand_dims(face, axis=0)
        yhat_class = clf.predict(sample)
        yhat_proba = clf.predict_proba(sample)
        index = yhat_class[0]
        probability = yhat_proba[0, index]
        if probability > 0.2:
            predict_name = lb.classes_[index]
        else:
            predict_name = 'Unknown'
        cv2.rectangle(img, (x1,y1), (x2, y2), (0, 255, 0), 1)
        x_text = x1
        y_text = y2 + 20
        cv2.putText(img, predict_name, (x_text, y_text), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1, (255, 0, 255), thickness=1, lineType=2)
        cv2.putText(img, str(round(probability, 3)), (x_text, y_text + 20),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), thickness=1, lineType=2)
    cv2.imshow('image', img)
    end = time.time()
    t = end - start
    print('Time for an image: %.3f s' % (t))
    cv2.imwrite('testImage3.jpg', img)
    k = cv2.waitKey() & 0xff
    if k == 27:
        break

'''  
    i = i+1
    print(i, face.shape)
    # Bieu dien
    plt.subplot(6,6,i)
    plt.axis('off')
    plt.imshow(face)
plt.show()
print(results)
'''