import numpy as np
import argparse
import cv2
import os
from model_keras import build_model
from parameters_keras import TRAINING, NETWORK

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug import secure_filename


model = build_model()
if NETWORK.output_size == 7:
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
elif NETWORK.output_size == 3:
    emotion_dict = {0: "Negative", 1: "Positive", 2: "Neutral"}
try:
    model.load_weights(TRAINING.save_model_path+'model.h5')
except:
    print('Failed to load weights from - ',TRAINING.save_model_path+'model.h5')
    exit()
    

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('uploaded_file',
                                filename=filename))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    PATH_TO_TEST_IMAGES_DIR = app.config['UPLOAD_FOLDER']
    TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR,filename.format(i)) for i in range(1, 2) ]
    IMAGE_SIZE = (12, 8)

    for image_path in TEST_IMAGE_PATHS:
        # prevents openCL usage and unnecessary logging messages
        cv2.ocl.setUseOpenCL(False)
        #
        frame = cv2.imread(image_path)
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
            cv2.imwrite('uploads/'+filename,frame)
            #cv2.imwrite('test_'+path+'.jpg',frame)
            return send_from_directory(app.config['UPLOAD_FOLDER'],filename)
            
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)