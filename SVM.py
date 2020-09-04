import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

print('........Loading.........')

#Chuan bi du lieu
data = np.load('backup/faces-embedding-train.npz')
X_train,y_train = data['arr_0'],data['arr_1']
data = np.load('backup/faces-embedding-test-1.npz')
X_test_1,y_test_1 = data['arr_0'],data['arr_1']
data = np.load('backup/faces-embedding-test-2.npz')
X_test_2,y_test_2 = data['arr_0'],data['arr_1']

#Xu ly du lieu
out_encoder = LabelEncoder()
out_encoder.fit(y_train)
y_train = out_encoder.transform(y_train)
y_test_1 = out_encoder.transform(y_test_1)
y_test_2 = out_encoder.transform(y_test_2)

#Train
model = SVC(kernel='linear',probability=True)
model.fit(X_train,y_train)
out = open("output/clf.pkl","wb")
out.write(pickle.dumps(model))
out.close()
out = open("output/lb.pkl","wb")
out.write(pickle.dumps(out_encoder))
out.close()

#Predict va Accuracy
ypred_train = model.predict(X_train)
ypred_test_1 = model.predict(X_test_1)
ypred_test_2 = model.predict(X_test_2)
acc_train = accuracy_score(y_train,ypred_train)*100
acc_test_1 = accuracy_score(y_test_1,ypred_test_1)*100
acc_test_2 = accuracy_score(y_test_2,ypred_test_2)*100
print('Accuracy: train = %.3f' %(acc_train))
print('Accuracy: test_1 = %.3f' %(acc_test_1))
print('Accuracy: test_2 = %.3f' %(acc_test_2))
print('Done!!!')