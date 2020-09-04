import os
import cv2
import uuid
import json
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
from flask import Flask, Response, request

os.environ["CUDA_VISIBLE_DEVICES"] = "O"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

tf.compat.v1.keras.backend.set_session(session)

#Phat hien khuon mat trong anhimport
detector = MTCNN()
def faceDetect(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(image)
    bbox = result[0]['box']
    cv2.rectangle(image, (bbox[0],bbox[1]), (bbox[0]+bbox[2],bbox[1]+bbox[3]),
                  (0,255,0), thickness=2)
    cv2.putText(image, 'face', (bbox[0], bbox[1]+bbox[3]+25), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), thickness=2)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    path_file = ('static/%s.jpg' %uuid.uuid4().hex)
    cv2.imwrite(path_file, image)
    return json.dumps(path_file)

#

#API
app = Flask(__name__)

#route http post to this method
@app.route('/api/upload', methods=['POST'])

#Main
def upload():
    image = cv2.imdecode(np.fromstring(request.files['image'].read(),np.uint8),
                         cv2.IMREAD_UNCHANGED)
    image = faceDetect(image)
    return Response(response=image, status=200, mimetype="application/json")

#Start server
app.run(host="0.0.0.0", port=5000)
