import cv2
from frame import frame


class faces():
    def __init__(self):
        self.cap = frame()
        self.cap.start()
        self.face_cascade = cv2.CascadeClassifier("FrontFaceCascade.xml")

    def detect_faces(self):
        frames = cv2.imread("img.jpg")
        faces = face_cascade.detectMultiScale(frames)
        for (x,y,w,h) in faces:
            img = cv2.rectangle(frames,(x,y),(w,h),(255,0,0),2)
        cv2.imshow("video",img)
        return img

    def destroy(self):
        self.cap.destroy()
        cv2.destroyAllWindows()
