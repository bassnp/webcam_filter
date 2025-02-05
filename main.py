from math import floor
import cv2
import torch
import math
import pyvirtualcam
from pyvirtualcam import PixelFormat
import numpy as np

def int_to_clr(n):
    r = int((math.sin(n) * 1000) % 256)
    g = int((math.cos(n) * 1000) % 256)
    b = int((math.tan(n) * 1000) % 256)
    return r, g, b

def blend_rect(frame, x, y, w, h, clr):
    sub_img = frame[y:y+h, x:x+w]
    shader_img  = np.full(sub_img.shape, clr, np.uint8)
    blend = cv2.addWeighted(sub_img, 0.35, shader_img, 0.35, 1.0)
    frame[y:y+h, x:x+w] = blend

def draw_ai_label(frame, txt, x, y, clr, conf):
    conf_perc = floor(conf * 10000) / 100
    conf_brightness = int(conf * 256)
    conf_blackwhite = (conf_brightness, conf_brightness, conf_brightness)
    cv2.putText(frame, txt, (int(x + 5), int(y - 10)), cv2.FONT_HERSHEY_DUPLEX, 0.8, clr if clr != -1 else conf_blackwhite, 1)
    cv2.putText(frame, str(conf_perc), (int(x + 5), int(y + 15)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, conf_blackwhite, 1)

def draw_ai_box(frame, box, txt, r, g, b):
    x, y, x2, y2, conf, cls = box
    x, y = int(x), int(y)
    w, h = int(x2-x), int(y2-y)
    clr = (r, g, b)

    blend_rect(frame, x, y, w, h, clr)
    draw_ai_label(frame, txt, x, y, clr, conf)

def ai_filter(frame, model):
    results = model(frame)
    boxes = results.xyxy[0].cpu().numpy()
    sightings = []

    for box in boxes:
        x1, y1, x2, y2, conf, cls = box

        name = model.names[int(cls)]
        r, g, b = int_to_clr(int(cls))

        if name not in sightings:
            sightings.append(name)

        draw_ai_box(frame, box, name, r, g, b)

    return frame, sightings

def hue_filter(frame, model):
    results = model(frame)
    boxes = results.xyxy[0].cpu().numpy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    dark_img  = np.full(image_gray.shape, (0, 0, 0), np.uint8)
    blend = cv2.addWeighted(image_gray, 0.4, dark_img, 0.4, 1.0)

    # Combine all hue blends
    for box in boxes:
        x, y, x2, y2, conf, cls = box
        x, y = int(x), int(y)
        w, h = int(x2-x), int(y2-y)
        blend[y:y+h, x:x+w] = frame[y:y+h, x:x+w]

    # Draw afterwards
    for box in boxes:
        x, y, x2, y2, conf, cls = box
        x, y = int(x), int(y)
        name = model.names[int(cls)]
        name = 'idiot' if name == 'person' else name
        draw_ai_label(blend, name, x, y, -1, conf)

    return blend

def main(filter_name):
    print(torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using " + str(device))

    # Image scanning model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model.to(device)

    # Webcam capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    # Virtual Camera
    with (pyvirtualcam.Camera(width=640, height=480, fps=60, fmt=PixelFormat.BGR, backend ='unitycapture') as cam):
        # Webcam Loop
        while cap.isOpened():
            ret, frame = cap.read() # Read webcam frame/image
            if ret is not None and frame is not None:
                # Switch desired filter name
                frame_filtered = None
                if filter_name == "ai":
                    frame_filtered, sightings = ai_filter(frame, model)
                elif filter_name == "hue":
                    frame_filtered = hue_filter(frame, model)
                else: # Default
                    frame_filtered = frame

                # Show window debug
                half_size_frame = cv2.resize(frame_filtered, (0, 0), fx=0.3, fy=0.3)
                cv2.imshow("webcam", half_size_frame)
                cv2.setWindowProperty("webcam", cv2.WND_PROP_TOPMOST, 1)

                # Send to virtual cam
                cam.send(frame_filtered)

            if cv2.waitKey(1) == ord('q'): # Quit on q
                break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main("hue")