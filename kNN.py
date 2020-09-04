import pickle
import numpy as np
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

data = np.load('backup/faces-embedding-train.npz')
X_train,y_train = data['arr_0'],data['arr_1']
data = np.load('backup/faces-embedding-test-1.npz')
X_test_1,y_test_1 = data['arr_0'],data['arr_1']
data = np.load('backup/faces-embedding-test-2.npz')
X_test_2,y_test_2 = data['arr_0'],data['arr_1']

out_encoder = LabelEncoder()
out_encoder.fit(y_train)
y_train = out_encoder.transform(y_train)
y_test_1 = out_encoder.transform(y_test_1)
y_test_2 = out_encoder.transform((y_test_2))

model = neighbors.KNeighborsClassifier(n_neighbors=3,weights='distance',p=2)
model.fit(X_train,y_train)
out = open('output/knn.pkl','wb')
out.write(pickle.dumps(model))
out.close()
out = open('output/knn_lb.pkl','wb')
out.write(pickle.dumps(out_encoder))
out.close()

y_pred_1 = model.predict(X_test_1)
y_pred_2 = model.predict(X_test_2)
acc_1 = accuracy_score(y_test_1,y_pred_1)*100
acc_2 = accuracy_score(y_test_2,y_pred_2)*100
print('Accuracy of test 1: %.3f' %(acc_1))
print('Accuracy of test 2: %.3f' %(acc_2))
print('Done!!!')
