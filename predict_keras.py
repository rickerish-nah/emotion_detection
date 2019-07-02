import numpy as np
import argparse
import cv2
import os
from model_keras import build_model
from parameters_keras import TRAINING, NETWORK

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode",help="video/image")
ap.add_argument("--path",default="test.jpg")
a = ap.parse_args()
mode = a.mode
path = a.path
print(mode,path)

# load model
model = build_model()

# dictionary which assigns each label an emotion (alphabetical order)
if NETWORK.output_size == 7:
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
elif NETWORK.output_size == 3:
    emotion_dict = {0: "Negative", 1: "Positive", 2: "Neutral"}

# emotions will be displayed on your face from the webcam feed
if mode == "video":
    
    try:
        model.load_weights(TRAINING.save_model_path+'model.h5')
    except:
        print('Failed to load weights from - ',TRAINING.save_model_path+'model.h5')
        exit()
    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)
    
    # start the webcam feed
    cap = cv2.VideoCapture(0)
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))

            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
        cv2.imshow('Video', cv2.resize(frame,(512,512),interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

elif mode == "image":
    try:
        model.load_weights(TRAINING.save_model_path+'model.h5')
    except:
        print('Failed to load weights from - ',TRAINING.save_model_path+'model.h5')
        exit()
    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)
    #
    frame = cv2.imread(path)
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)

        maxindex = int(np.argmax(prediction))
        #print(maxindex,'---------------->>>>>>>')
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        #cv2.imshow('Image', cv2.resize(frame,(600,512),interpolation = cv2.INTER_CUBIC))
        cv2.imwrite('test_op.jpg',frame)


