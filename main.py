import os
import cv2
import uuid
import json
import pickle
import numpy as np
from mtcnn import MTCNN
from random import choice
import matplotlib.pyplot as plt
from keras.models import load_model
from flask import Flask,Response,request
from processing import Detect,get_embedding

#Chuan bi model
detector = MTCNN()
clf = pickle.loads(open('output/knn.pkl','rb').read())
lb = pickle.loads(open('output/knn_lb.pkl','rb').read())
model = load_model('model/facenet_keras.h5',compile=False)

#Hien thi ket qua
def show(image,name,p):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if p == None:
        plt.axis('off')
        plt.imshow(image)
        plt.title(name)
    else:
        dir = 'data/train/' + name + '/'
        filename = choice([filename for filename in os.listdir(dir)])
        path = dir + filename
        inData = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        #cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),thickness=2)
        plt.subplot(1,2,1)
        plt.axis('off')
        plt.imshow(image)
        plt.title(name+'\n'+str(p)+'%')
        plt.subplot(1,2,2)
        plt.axis('off')
        plt.imshow(inData)
        plt.title('Sample in Data')
    path_file = ('static/%s.jpg' % uuid.uuid4().hex)
    #cv2.imwrite(path_file,image)
    #return image
    plt.savefig(path_file)
    return path_file

#Xu ly chinh
def predict(image):
    w,h = image.shape[1],image.shape[0]
    x1,y1,x2,y2,face,flags = Detect(image,detector=detector)
    if flags == False:
        p = None
        if (x1+y1+x2+y2) == 0:
            name = 'Detected No Face!!!'
        elif (x1+y1+x2+y2) == 4:
            name = 'Detected More Than 2 Faces!!!'
    else:
        sample = get_embedding(face,model=model)
        sample = np.expand_dims(sample,axis=0)
        y_class = clf.predict(sample)[0]
        y_proba = clf.predict_proba(sample)
        p = y_proba[0,y_class]
        if p>0.6:
            p = round(p*100,2)
            name = lb.classes_[y_class]
        else:
            p = None
            name = 'Not Include In Data'
    path_file = show(image,name,p)
    return json.dumps(path_file)
'''
image = cv2.imread('Data/val/Alejandro_Toledo/Alejandro_Toledo_0033.jpg',
                   cv2.IMREAD_UNCHANGED)
image = predict(image)
cv2.imshow('1',image)
cv2.waitKey(0)
'''
#API
app = Flask(__name__)

#route http post to this method
@app.route('/api/upload',methods=['POST'])

#Main

def upload():
    image = cv2.imdecode(np.fromstring(request.files['image'].read(), np.uint8),
                         cv2.IMREAD_UNCHANGED)
    image = predict(image)
    return Response(response=image, status=200, mimetype="application/json")

#Start server
app.run(host="0.0.0.0", port=5000)