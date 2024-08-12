import time
import cv2
import torch
from depthDetection.misc import colorize

model = torch.hub.load("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("Using CPU")

midas = model.to(DEVICE)


def get_depth(frame):
    start_time = time.time()
    frame = torch.from_numpy(frame).to(DEVICE)
    frame = frame.unsqueeze(0)  # add batch dim
    depth = midas(frame)  # might need to adjust depending on model input
    colored = colorize(depth)
    out_frame = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
    end_time = time.time()
    if end_time - start_time > 0.1:
        print("FPS: ", 1 / (end_time - start_time))
    return out_frame, depth
