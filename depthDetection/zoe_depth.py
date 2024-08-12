import os

import PIL.Image
import cv2
import numpy as np
import torch

from multiprocessing import Process
from depthDetection.misc import colorize


def get_depth(in_file):
    print("Loading ZoeDepth")
    # torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh download of MiDaS repo
    repo = "isl-org/ZoeDepth"
    model = "ZoeD_NK"
    model_zoe = torch.hub.load(repo, model, pretrained=True)

    # If "Using CPU", check this: https://pytorch.org/get-started/locally/
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cpu":
        print("Using CPU")

    zoe = model_zoe.to(DEVICE)
    cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)

    while True:
        if os.path.exists(in_file) is False:
            continue

        nd_arr = np.fromfile(in_file, np.uint8)
        bytearr = nd_arr.tobytes()
        try:
            frame = PIL.Image.frombytes("RGB", (640, 480), bytearr)
        except:
            continue

        if frame is None:
            print(f"Could not read the file {in_file}. Check that the file exists and the path is correct.")
        else:
            depth = zoe.infer_pil(frame)
            colored = colorize(depth)
            out_frame = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
            cv2.imshow("Depth", out_frame)
            cv2.waitKey(1) & 0xFF
            os.remove(in_file)


def create_depth_thread(in_file):
    depth_thread = Process(target=get_depth, daemon=True, args=(in_file,))
    depth_thread.start()
    print("ZoeDepth thread started")
