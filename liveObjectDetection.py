import os
from multiprocessing import Process

import cv2

from audioRecording import create_audio_thread
from intentRecognition.intentRecognizer import recognize_intent
from objectDetection import detect_objects, load_weights, parse_objects
from objectDetection.objectDetection import draw_text

cwd = os.getcwd()
folder = "\\files"
transcript_file = cwd + folder + "\\transcript.txt"
if not os.path.exists(cwd + folder):
    os.makedirs(cwd + folder)


def main():
    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture(0)
    print("[INFO] video stream started")
    last_object_count = 0
    transcript = ""
    load_weights()
    if os.path.exists(transcript_file):
        os.remove(transcript_file)
    with open(transcript_file, "a", encoding="utf-8") as f:
        f.write("")
    audio_thread = create_audio_thread(transcript_file)
    print("[INFO] Ready.")

    while True:
        (grabbed, frame) = vs.read()

        if not grabbed:
            break

        boxes, confidences, indices = detect_objects(frame)
        current_objects = parse_objects(boxes, confidences, indices)

        if os.path.exists(transcript_file):
            with open(transcript_file, "r", encoding="utf-8") as f:
                full_transcript = f.read()
                lines = full_transcript.split("\n")
                if len(lines) > 1:
                    last_line = lines[-2]
                    if last_line.strip() != "" and last_line.strip() != transcript:
                        transcript = last_line.strip()
                        Process(target=recognize_intent, daemon=True, args=(transcript, current_objects, )).start()
        else:
            print("[INFO] No transcript file found")

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
    cv2.destroyAllWindows()
    audio_thread.join()


if __name__ == "__main__":
    main()
