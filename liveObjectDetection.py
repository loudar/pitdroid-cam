import os

import cv2

from audioRecording.audioRecordingTest import create_audio_thread
from objectDetection import detect_objects, load_weights
from objectDetection.objectDetection import draw_text

cwd = os.getcwd()
transcript_file = cwd + "/transcript.txt"


def main():
    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture(0)
    print("[INFO] video stream started")
    last_object_count = 0
    current_audio_thread = None
    transcript = ""
    load_weights()

    while True:
        (grabbed, frame) = vs.read()

        if not grabbed:
            break

        boxes, confidences, indices = detect_objects(frame)
        # show_objects(frame, boxes, indices, last_object_count)

        if current_audio_thread is None or current_audio_thread.is_alive() is False:
            current_audio_thread = create_audio_thread(transcript_file)

            if os.path.exists(transcript_file):
                with open(transcript_file, "r") as f:
                    transcript = f.read()

        draw_text(frame, 0, transcript, 10, 10, (255, 255, 255))
        cv2.imshow("Objects", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord(" "):
            while True:
                key = cv2.waitKey(1) or 0xff
                if key == ord(" "):
                    break

    print("[INFO] cleaning up...")
    vs.release()


if __name__ == "__main__":
    main()


def show_objects(frame, boxes, indices, last_object_count):
    if len(indices) < last_object_count:
        print(f"Detected {len(indices)} objects")
        for j in range(len(indices), last_object_count):
            cv2.destroyWindow(f"Object {j}")

    for i in indices:
        try:
            box = boxes[i]
        except:
            i = i[0]
            box = boxes[i]

        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        cropped_frame = frame[int(y):int(y + h), int(x):int(x + w)]
        if cropped_frame.size != 0:
            cv2.imshow(f"Object {i}", cropped_frame)
        else:
            print(f"Object {i} has empty bounding box")

    last_object_count = len(indices)
