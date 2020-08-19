import numpy as np 
import cv2 as cv
import time
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt


PROTO_PATH = "deploy.prototxt"
MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"

FPS = 15
WINDOW_TIME_SEC = 30
WINDOW_SIZE = int(np.ceil(WINDOW_TIME_SEC * FPS))
MIN_HR_BPM = 45.0
MAX_HR_BMP = 240.0
MAX_HR_CHANGE = 12.0
SEC_PER_MIN = 60

detector = cv.dnn.readNetFromCaffe(PROTO_PATH,MODEL_PATH)
video = cv.VideoCapture(0)
time.sleep(2)

buffer = []
start = time.time()


while(time.time() - start < 360):
    _,frame = video.read()

    (h, w) = frame.shape[:2]

    image_blob = cv.dnn.blobFromImage(
            cv.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)
    detector.setInput(image_blob)
    detections = detector.forward()

    index = np.argmax(detections[0,0,:,2])
    box = detections[0,0,index,3:7] * np.array([w,h,w,h]) 
    x1,y1,x2,y2 = box.astype(int)

    face_box = frame[y1:y2,x1:x2]

    col = face_box.reshape(-1, face_box.shape[-1])
    buffer.append(col.mean(axis = 0))


    if(time.time() - start > 30):
        mean = np.mean(buffer[-450:])
        std = np.std(buffer[-450:])
        normalized = (buffer[-450:] - mean ) / std

        ica = FastICA()
        signal = ica.fit_transform(normalized)
        ps = np.abs(np.fft.fft(signal, axis=0))**2
        freqs = np.fft.fftfreq(len(buffer[-450:]), 1.0 / FPS)


        maxPwrSrc = np.max(ps, axis=1)
        validIdx = np.where((freqs >= MIN_HR_BPM / SEC_PER_MIN) & (freqs <= MAX_HR_BMP / SEC_PER_MIN))
        validPwr = maxPwrSrc[validIdx]
        validFreqs = freqs[validIdx]
        maxPwrIdx = np.argmax(validPwr)
        hr = validFreqs[maxPwrIdx]
        print(hr * 60)

video.release()
cv.destroyAllWindows()