import numpy as np
import pickle
import time
import os
import matplotlib.pyplot as plt
from random import choice
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

#load faces
data = np.load('backup/celebrity-faces-test.npz')
testX_faces = data['arr_0']

start = time.time()
# load dataset
data = np.load('backup/celebrity-faces-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))
# chuan hoa vector dau vao
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
print(trainX[1])

testX = in_encoder.transform(testX)
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)
# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)
# Luu model
out = open("output/clf.pkl", "wb")
out.write(pickle.dumps(model))
out.close()
out = open("output/lb.pkl", "wb")
out.write(pickle.dumps(out_encoder))
out.close()

# predict
ypred_train = model.predict(trainX)
ypred_test = model.predict(testX)
# Accuracy
acc_train = accuracy_score(trainy, ypred_train)*100
acc_test = accuracy_score(testy, ypred_test)*100
print('Accuracy: train=%.3f, test=%.3f' % (acc_train, acc_test))
end = time.time()
print('Train Model Time: %.3f s' %(end - start))
'''
# test model on a random example from the test dataset
selection = choice([i for i in range(testX.shape[0])])
random_face_pixels = testX_faces[selection]
random_face_emb = testX[selection]
random_face_class = testy[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])
# prediction for the face
samples = np.expand_dims(random_face_emb, axis=0)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)
# get name
class_index = yhat_class[0]
class_probability = yhat_prob[0,class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)
print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
print('Expected: %s' % random_face_name[0])
print('Max Prob: %s' % (np.argmax(yhat_prob[0,:])))
# plot
plt.imshow(random_face_pixels)
title = '%s (%.3f)' % (predict_names[0], class_probability)
plt.title(title)
plt.show()
'''

import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims = True)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
'''
cnf_matrix = confusion_matrix(testy, ypred_test)
class_names = []
for subdir in os.listdir('5-celebrity-faces-dataset/val'):
    class_names.append(subdir)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.show()
'''