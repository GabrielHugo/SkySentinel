import cv2
import numpy as np

class MotionDetector :

    def __init__(self):

        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=250, detectShadows=False)

    def detect(self, frame) :

        # Filter

        blur = cv2.GaussianBlur(frame, (15, 15), 0)

        fgmask = self.fgbg.apply(blur)

        _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

        dilated = cv2.dilate(thresh, None, iterations=2)


        # Bounding box

        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours :
            cnt_area = cv2.contourArea(contour)

            if cnt_area < 1000 or cnt_area > 4000 :
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))
            print(cnt_area)

        fgmask_bgr = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
        out_frame = cv2.hconcat([frame, fgmask_bgr])

        return out_frame
