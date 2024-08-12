import time

import cv2
import torch

from depthDetection.misc import colorize

torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh download of MiDaS repo
repo = "isl-org/ZoeDepth"
model = "ZoeD_NK"
# Zoe_N
model_zoe = torch.hub.load(repo, model, pretrained=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("Using CPU")

zoe = model_zoe.to(DEVICE)


def get_depth(frame):
    start_time = time.time()
    depth = zoe.infer_pil(frame)

    colored = colorize(depth)
    out_frame = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
    end_time = time.time()
    if end_time - start_time > 0.1:
        print("FPS: ", 1 / (end_time - start_time))

    return out_frame, depth
