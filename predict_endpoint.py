import numpy as np
import argparse
import cv2
import os
from model_keras import build_model
from parameters_keras import TRAINING, NETWORK

class emotion_detection():
    def __init__(self):
        super(emotion_detection, self).__init__()

        self._build()

    def _build(self):
        self.model = build_model()
        if NETWORK.output_size == 7:
            self.emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
        elif NETWORK.output_size == 3:
            self.emotion_dict = {0: "Negative", 1: "Positive", 2: "Neutral"}
        try:
            self.model.load_weights(TRAINING.save_model_path+'model.h5')
        except:
            print('Failed to load weights from - ',TRAINING.save_model_path+'model.h5')
            exit()
    
    def predict_image(self,path = None):
        if path == None:
            print('load image please')
        else:
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
                prediction = self.model.predict(cropped_img)

                maxindex = int(np.argmax(prediction))
                #print(maxindex,'---------------->>>>>>>')
                cv2.putText(frame, self.emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                #cv2.imshow('Image', cv2.resize(frame,(600,512),interpolation = cv2.INTER_CUBIC))
                cv2.imwrite('test_'+path+'.jpg',frame)
                return frame



