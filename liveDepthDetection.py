import cv2

from objectDetection import detect_objects


def main():
    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture(0)
    print("[INFO] video stream started")

    while True:
        (grabbed, frame) = vs.read()

        if not grabbed:
            break

        boxes, confidences, indices = detect_objects(frame)
        print(f"Found {len(boxes)} objects")
        for box, confidence, index in zip(boxes, confidences, indices):
            cropped_frame = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            cv2.imshow(f"Object {index}", cropped_frame)

        cv2.imshow("Objects", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    print("[INFO] cleaning up...")
    vs.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
