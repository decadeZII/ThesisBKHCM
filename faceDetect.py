import os
import cv2
import numpy as np
from mtcnn import MTCNN
from processing import Detect

detector = MTCNN()

print('...Loading...')

#Load anh
def load_faces(dir):
    faces = list()
    for filename in os.listdir(dir):
        path =  dir + filename
        image = cv2.imread(path,cv2.IMREAD_UNCHANGED)
        _,_,_,_,face,_ = Detect(image,detector=detector)
        faces.append(face)
    return faces
def load_dataset(dir):
    X,y = list(),list()
    for subdir in os.listdir(dir):
        path = dir + subdir + "/"
        if not os.path.isdir(path):
            continue
        faces = load_faces(path)
        labels = [subdir for _  in range(len(faces))]
        print('>load %d image for : %s' % (len(faces),subdir))
        X.extend(faces)
        y.extend(labels)
    return np.asarray(X),np.asarray(y)

#Tao va luu du lieu
X_train,y_train = load_dataset('data/train/')
X_test_1, y_test_1 = load_dataset('data/val/')
X_test_2, y_test_2 = load_dataset('data/flash_val/')
print('Number of class: %.3f' %(y_train.shape))
print('Number of trained image: %.3f'%(y_test_1.shape))
print('Number of tested image: %.3f' %(y_test_2.shape))

np.savez_compressed('backup/train',X_train,y_train)
np.savez_compressed('backup/test_1',X_test_1,y_test_1)
np.savez_compressed('backup/test_2',X_test_2,y_test_2)
print('Done!')