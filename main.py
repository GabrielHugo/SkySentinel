import cv2
from camera_handler import SkyCamera
from motion_detector import MotionDetector

if __name__ == '__main__':

    camera = SkyCamera()
    detecteur = MotionDetector()

    while True :

        success, frame = camera.get_frame()

        img_traitee = detecteur.detect(frame)

        cv2.imshow("Caméra", img_traitee)

        if cv2.waitKey(1) == ord("q") :
            break

    camera.release_camera()

