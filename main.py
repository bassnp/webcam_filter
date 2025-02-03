import datetime
from math import floor

import cv2
import torch
import math
import pyvirtualcam
from pyvirtualcam import PixelFormat
import matplotlib.pyplot as plt
import numpy as np

def int_to_clr(n):
    r = int((math.sin(n) * 1000) % 256)
    g = int((math.cos(n) * 1000) % 256)
    b = int((math.tan(n) * 1000) % 256)
    return r, g, b

def draw_box(frame, box, txt, r, g, b):
    x, y, x2, y2, conf, cls = box
    x, y = int(x), int(y)
    w, h = int(x2-x), int(y2-y)

    clr = (r, g, b)
    conf_clr = (int(r * conf), int(g * conf), int(b * conf))
    conf_perc = floor(conf * 10000) / 100
    conf_brightness = int(conf * 256)
    conf_blackwhite = (conf_brightness, conf_brightness, conf_brightness)

    sub_img = frame[y:y+h, x:x+w]
    shader_img  = np.full(sub_img.shape, clr, np.uint8)
    blend = cv2.addWeighted(sub_img, 0.35, shader_img, 0.35, 1.0)
    frame[y:y+h, x:x+w] = blend

    cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), clr, 2)
    cv2.putText(frame, txt, (int(x + 5), int(y - 10)), cv2.FONT_HERSHEY_DUPLEX, 0.8, clr, 1)
    cv2.putText(frame, str(conf_perc), (int(x + 5), int(y + 15)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, conf_blackwhite, 1)

def scan_frame(tlist, frame, model):
    results = model(frame)
    boxes = results.xyxy[0].cpu().numpy()
    sightings = []

    for box in boxes:
        x1, y1, x2, y2, conf, cls = box

        name = model.names[int(cls)]
        r, g, b = int_to_clr(int(cls))

        if name not in sightings: # Quick log
            sightings.append(name)

        draw_box(frame, box, name, r, g, b)

    return frame, sightings

def main(track_list = [], capture_list = []):
    print(torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using " + str(device))

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model.to(device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    last_sighting = None
    print("Setupcam " + str(device))
    with pyvirtualcam.Camera(width=640, height=480, fps=60, fmt=PixelFormat.BGR, backend ='unitycapture') as cam:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is not None and frame is not None:
                # Scan Image
                named_frame, sightings = scan_frame(track_list, frame, model)

                # Show window
                half_size_frame = cv2.resize(named_frame, (0, 0), fx=0.3, fy=0.3)
                cv2.imshow("webcam", half_size_frame)
                cv2.setWindowProperty("webcam", cv2.WND_PROP_TOPMOST, 1)

                cv2.moveWindow("webcam", 50, 35)

                # Send to virtual cam
                cam.send(named_frame)

            if cv2.waitKey(1) == ord('q'): # Quit on q
                break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(
        # Specific tracking
        [],

        # Specific capturing
        ["bird"]
    )