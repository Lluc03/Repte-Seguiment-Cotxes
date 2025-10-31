import cv2

class VideoReader:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)

    def read(self):
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        self.cap.release()