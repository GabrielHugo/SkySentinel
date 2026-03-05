import cv2
from camera_handler import SkyCamera

class MotionDetector :

    def __init__(self):
        camera = SkyCamera()

        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=30, detectShadows=False)

    def detect(self, frame) :

        blur = cv2.GaussianBlur(frame, (7, 7), 0)

        fgmask = self.fgbg.apply(blur)

        fgmask_bgr = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
        out_frame = cv2.hconcat([blur, fgmask_bgr])

        return out_frame
