import numpy as np
from processing import get_embedding
from keras.models import load_model

model = load_model('model/facenet_keras.h5',compile=False)

print('......Loading......')

train = np.load('backup/train.npz')
X_train,y_train = train['arr_0'],train['arr_1']
test_1 = np.load('backup/test_1.npz')
X_test_1,y_test_1 = test_1['arr_0'],test_1['arr_1']
test_2 = np.load('backup/test_2.npz')
X_test_2,y_test_2 = test_2['arr_0'],test_2['arr_1']
print('Loaded: ',X_train.shape,y_train.shape)
print('Loaded: ',X_test_1.shape,y_test_1.shape)
print('Loaded: ',X_test_2.shape,y_test_2.shape)

#Lam viec voi tap train
newX_train = list()
for face_pixels in X_train:
    embedding = get_embedding(face_pixels,model)
    newX_train.append(embedding)
newX_train = np.asarray(newX_train)
#print(newX_train.shape)

#Lam viec voi tap test
#'''
newX_test_1 = list()
for face_pixels in X_test_1:
    embedding = get_embedding(face_pixels,model)
    newX_test_1.append(embedding)
newX_test_1 = np.asarray(newX_test_1)
#print(newX_test_1.shape)
newX_test_2 = list()
for face_pixels in X_test_2:
    embedding = get_embedding(face_pixels,model)
    newX_test_2.append(embedding)
newX_test_2 = np.asarray(newX_test_2)
#print(newX_test_2.shape)
#'''

np.savez_compressed('backup/faces-embedding-train',newX_train,y_train)
np.savez_compressed('backup/faces-embedding-test-1',newX_test_1,y_test_1)
np.savez_compressed('backup/faces-embedding-test-2',newX_test_2,y_test_2)
print('Done!!')
