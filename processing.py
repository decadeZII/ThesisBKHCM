import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
#'''
os.environ["CUDA_VISIBLE_DEVICES"] = "O"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session=session)
tf.function(experimental_relax_shapes=True)
#'''
#Tach khuon mat trong anh
def Detect(image,detector):
    # Tao bo phat hien khuon mat, su dung trong so mac dinh
    num_faces = 0
    img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    img = np.asarray(img)
    result = detector.detect_faces(img)
    for face in result:
        num_faces+=1
    if num_faces>0:
        if num_faces>1:
            x1,y1,x2,y2,face,flags = 1,1,1,1,None,False
        else:
            x1,y1,w,h = result[0]['box']
            # Truong hop tra ve gia tri am
            x1, y1 = abs(x1), abs(y1)
            x2,y2 = x1+w,y1+h
            face = img[y1:y2,x1:x2]
            face = Image.fromarray(face)
            face = face.resize((160,160))
            face = np.asarray(face).astype('float32')
            flags = True
    else:
        x1,y1,x2,y2,face,flags = 0,0,0,0,None,False
    return x1,y1,x2,y2,face,flags

def normalization(vector):
    a = 0
    for i in range(len(vector)):
        a = np.sqrt(a**2 + vector[i]**2)
    for i in range(len(vector)):
        vector[i] = vector[i] / a
    return vector

def get_embedding(face_pixels,model):
    mean,std = face_pixels.mean(),face_pixels.std()
    face_pixels = (face_pixels - mean)/std
    sample = np.expand_dims(face_pixels,axis=0)
    yhat = model.predict(sample)[0]
    yhat = np.asarray(yhat)
    #embedding = normalization(yhat)
    return yhat