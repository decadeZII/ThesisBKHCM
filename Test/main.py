import cv2
import numpy as np
import imutils
import pickle
from imutils.video import VideoStream
from PIL import Image
from mtcnn.mtcnn import MTCNN
from keras.models import load_model

# Load models
detector = MTCNN()
extractor = load_model('facenet_keras.h5')
clf = pickle.loads(open('output/clf.pkl', "rb").read())
lb = pickle.loads(open('output/lb.pkl', "rb").read())

def faceProcessing(face, size=(160, 160)):
    face = Image.fromarray(face)
    face = face.resize(size)
    face = np.asarray(face).astype('float32')
    mean, std = face.mean(), face.std()
    face_pixels = (face - mean) / std
    sample = np.expand_dims(face_pixels, axis=0)
    return sample

def normalization(vector):
    a = 0
    for i in range(len(vector)):
        a = np.sqrt(a**2 + vector[i]**2)
    for i in range(len(vector)):
        vector[i] = vector[i] / a
    return vector

def featureExtraction(face):
    faceEmbedding = extractor.predict(face)[0]
    faceEmbedding = np.asarray(faceEmbedding)
    faceEmbedding = normalization(faceEmbedding)
    sample = np.expand_dims(faceEmbedding, axis=0)
    return sample

def main():
    cap = VideoStream(src=0).start()
    while (True):
        frame = cap.read()
        frame = imutils.resize(frame, width=600)
        frame = cv2.flip(frame, 1)
        image = np.asarray(frame)
        faceDetect = detector.detect_faces(image)
        n = len(faceDetect)
        try:
            if n >= 1:
                for i in range(len(faceDetect)):
                    x1, y1, width, weight = faceDetect[i]['box']
                    x1, y1 = abs(x1), abs(y1)
                    x2, y2 = x1 + width, y1 + weight
                    face = image[y1:y2, x1:x2]
                    face = faceProcessing(face)
                    sample = featureExtraction(face)
                    y_class = clf.predict(sample)[0]
                    y_proba = clf.predict_proba(sample)
                    p = round(y_proba[0, y_class], 3)
                    if p < 0.1:
                        predict_name = 'Unknown'
                    else:
                        predict_name = lb.classes_[y_class]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2, lineType=2)
                    cv2.putText(frame, predict_name, (x1, y2 + 20), cv2.FONT_HERSHEY_COMPLEX, 1,
                                (255, 255, 255), thickness=1, lineType=2)
                    cv2.putText(frame, str(p), (x1, y2 + 45), cv2.FONT_HERSHEY_COMPLEX, 1,
                                (255, 255, 255), thickness=1, lineType=2)
            else:
                text = 'Detected no face'
                cv2.putText(frame, text, (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                            (255, 255, 255), thickness=1, lineType=2)
        except:
            pass
        cv2.imshow('result', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cv2.destroyAllWindows()

main()