import cv2

from objectDetection import detect_objects


def main():
    # initialize the video stream
    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture(0)

    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()

        if not grabbed:
            break

        detect_objects(frame)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    # release the file pointers
    print("[INFO] cleaning up...")
    vs.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
