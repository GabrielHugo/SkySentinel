import cv2
import os
from dotenv import load_dotenv

load_dotenv()

class SkyCamera :

    def __init__(self) :
        capture = os.getenv("MP4_2")
        self.camera = cv2.VideoCapture(capture)

    def get_frame(self) :
        ret, frame = self.camera.read()

        return ret, frame

    def release_camera(self) :
        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__" :
    camera = SkyCamera()

    while True :
        success, img = camera.get_frame()

        if not success :
            print("Erreur de lecture")
            break

        cv2.imshow( "Capture",img)

        if cv2.waitKey(1) == ord("q") :
            break

    camera.release_camera()