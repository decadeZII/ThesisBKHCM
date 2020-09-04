import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
import time

# Cat khuon mat tu buc anh dau vao
def extract_face(image, required_size=(160, 160)):
    # chuyen sang he mau RGB neu can
    image = image.convert('RGB')
    # chuyen du lieu thanh mang (array)
    pixels = np.asarray(image)
    # tao bo phat hien khuon mat, su dung trong so mac dinh
    detector = MTCNN()
    # phat hien khuon mat trong buc anh
    results = detector.detect_faces(pixels)
    x1, y1, width, height = results[0]['box']
    # phong truong hop tra ve gia tri am
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # cat khuon mat
    face = pixels[y1:y2, x1:x2]
    # chinh kich co anh cho phu hop model facenet
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array

# The hien cac khuon mat sau khi cat trong mot thu muc bat ky
def visuallize(folder, i=1):
    #liet ke cac tap tin
    for filename in os.listdir(folder):
        # duong dan
        path = folder + filename
        # lay khuon mat tu anh
        image = Image.open(path)
        face = np.asarray(image)
        #face = extract_face(image)
        print(i, face.shape)
        # plot
        plt.subplot(5,10,i)
        plt.axis('off')
        plt.imshow(face)
        i +=1
    plt.show()

# load anh va trich xuat khuon mat cho tat ca anh trong thu muc
def load_faces(directory):
    faces = list()
    for filename in os.listdir(directory):
        # duong dan
        path = directory + filename
        # trich xuat khuon mat
        image = Image.open(path)
        face = extract_face(image)
        faces.append(face)
    return faces

# load anh va trich xuat khuon mat trong tap du lieu
def load_dataset(directory):
    X, y = list(), list()
    for subdir in os.listdir(directory):
        # duong dan
        path = directory + subdir + '/'
        if not os.path.isdir(path):
            continue
        # trich xuat khuon mat trong thu muc con
        faces = load_faces(path)
        labels = [subdir for _ in range(len(faces))]
        print('>load %d image for : %s' % (len(faces), subdir))
        # luu lai du lieu
        X.extend(faces)
        y.extend(labels)
    return np.asarray(X), np.asarray(y)

# load train dataset
start = time.time()
trainX, trainy = load_dataset('5-celebrity-faces-dataset/train/')
print(trainX.shape, trainy.shape)
#load test dataset
#testX, testy = load_dataset('5-celebrity-faces-dataset/val/')
#print(testX.shape, testy.shape)
# luu vao file nen
np.savez_compressed('backup/celebrity-faces-train', trainX, trainy)
#np.savez_compressed('backup/celebrity-faces-test', testX, testy)
end = time.time()
print('detectFaces time: %.3f s' % (end - start))
#visuallize('5-celebrity-faces-dataset/train/David_Beckham/')