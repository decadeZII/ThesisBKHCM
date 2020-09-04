import numpy as np
from keras.models import load_model
import time

# tao vector dac trung cho mot khuon mat
def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    # chuan hoa cac gia tri pixels
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean)/std
    samples = np.expand_dims(face_pixels, axis=0)
    yhat = model.predict(samples)
    return yhat[0]

start = time.time()
# load tap du lieu
data_train = np.load('backup/celebrity-faces-train.npz')
data_test = np.load('backup/celebrity-faces-test.npz')
trainX, trainy = data_train['arr_0'], data_train['arr_1']
testX, testy = data_test['arr_0'], data_test['arr_1']
print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
# load facenet model
model = load_model('facenet_keras.h5')
print('Loaded Model')
# Lam viec voi tap train
newTrainX = list()
for face_pixels in trainX:
    embedding = get_embedding(model, face_pixels)
    newTrainX.append(embedding)
newTrainX = np.asarray(newTrainX)
print(newTrainX.shape)
# Lam viec voi tap test
newTestX = list()
for face_pixels in testX:
    embedding = get_embedding(model, face_pixels)
    newTestX.append(embedding)
newTestX = np.asarray(newTestX)
print(newTestX.shape)
# Luu vao file nen npz
np.savez_compressed('backup/celebrity-faces-embeddings', newTrainX, trainy, newTestX, testy)
end = time.time()
print('CreateFaceEmbeddings Time: %.3f s' %(end - start))
