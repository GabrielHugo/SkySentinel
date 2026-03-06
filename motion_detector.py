import cv2
import numpy as np
from camera_handler import SkyCamera

class MotionDetector :

    def __init__(self):

        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=250, detectShadows=False)

    def detect(self, frame) :

        # Filter

        contrast = 2
        brightness = 70

        frame[:, :, 2] = np.clip(contrast * frame[:, :, 2] + brightness, 0, 255)

        blur = cv2.GaussianBlur(frame, (7, 7), 0)

        normalize = cv2.normalize(frame, blur, 0, 255, cv2.NORM_MINMAX)

        fgmask = self.fgbg.apply(blur, normalize)

        # Bounding box

        kernel = np.ones((10, 10), np.uint8)
        dilation = cv2.dilate(fgmask, kernel, iterations=1)
        erosion = cv2.erode(fgmask, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i in range(0, len(contours)) :
            if i % 1 == 0 :
                cnt = contours[i]

                x, y, w, h = cv2.boundingRect(cnt)
                cv2.drawContours(fgmask, contours, -1, (255, 255, 0), 3)
                cv2.rectangle(fgmask, (x, y), (x + w, y + h), (255, 0, 0), 2)

        fgmask_bgr = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
        out_frame = cv2.hconcat([normalize, fgmask_bgr])


        return out_frame
