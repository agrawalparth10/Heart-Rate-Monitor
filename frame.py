import cv2
import time

class frame():
	def __init__(self):
		self.cap = None
		
	def start(self):
		print(f"[Camera is Starting]")
		self.cap = cv2.VideoCapture(0)
		time.sleep(2)
		return
		

	def capture_frame(self):
		_,cap_frame = self.cap.read()
		return cap_frame

	def destroy(self):
		print(f"[Camera is Shutting Down]")
		self.cap.release()
		return

