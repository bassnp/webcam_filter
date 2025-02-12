#2/4/25
from math import floor
import cv2
import torch
import math
import pyvirtualcam
from pyvirtualcam import PixelFormat
import numpy as np

# Load classifiers
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def int_to_clr(n):
    seed = 390
    r = int((math.sin(n) * seed) % 256)
    g = int((math.cos(n) * seed) % 256)
    b = int((math.tan(n) * seed) % 256)
    return r, g, b

def blend_rect(frame, x, y, w, h, clr):
    sub_img = frame[y:y+h, x:x+w]
    shader_img  = np.full(sub_img.shape, clr, np.uint8)
    blend = cv2.addWeighted(sub_img, 0.35, shader_img, 0.35, 1.0)
    frame[y:y+h, x:x+w] = blend

def outline_text(frame, txt, x, y, clr, size, thickness):
    cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_TRIPLEX, size, (0, 0, 0), thickness + 2)
    cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_TRIPLEX, size, clr, thickness)

def draw_ai_label(frame, txt, x, y, clr, conf):
    conf_perc = floor(conf * 10000) / 100
    conf_brightness = int(conf * 256)
    conf_blackwhite = (conf_brightness, conf_brightness, conf_brightness)
    outline_text(frame, txt, x + 1, y - 8, clr if clr != -1 else conf_blackwhite, 0.8, 1)
    outline_text(frame, str(conf_perc), x + 3, y + 12, clr if clr != -1 else conf_blackwhite, 0.4, 1)

def draw_ai_box(frame, box, txt, r, g, b):
    x, y, x2, y2, conf, cls = box
    x, y = int(x), int(y)
    w, h = int(x2-x), int(y2-y)
    clr = (r, g, b)
    blend_rect(frame, x, y, w, h, clr)
    draw_ai_label(frame, txt, x, y, clr, conf)

def ai_filter(frame, boxes, model):
    sightings = []

    for box in boxes:
        x1, y1, x2, y2, conf, cls = box

        name = model.names[int(cls)]
        r, g, b = int_to_clr(int(cls))

        if name not in sightings:
            sightings.append(name)

        draw_ai_box(frame, box, name, r, g, b)
    return frame, sightings

def deepfry_filter(frame, boxes, model):
    # Init frame filters
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    dark_img = cv2.convertScaleAbs(image_gray, alpha=0.5, beta=0)
    invert_img = cv2.bitwise_not(frame)
    saturated_image = cv2.convertScaleAbs(frame, alpha=1.2, beta=0)

    use_frame = dark_img

    # 1. Invert color of things first
    for (x1, y1, x2, y2, conf, cls) in boxes:
        if model.names[int(cls)] != 'person':
            x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
            use_frame[y:y+h, x:x+w] = invert_img[y:y+h, x:x+w]

    # 2. Select persons only
    for (x1, y1, x2, y2, conf, cls) in boxes:
        if model.names[int(cls)] == 'person':
            x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)

            person = frame[y:y+h, x:x+w]
            use_frame[y:y+h, x:x+w] = saturated_image[y:y+h, x:x+w]

            # Detection via cascade classifiers
            faces = face_cascade.detectMultiScale(person, 1.3, 5)
            for (fx, fy, fw, fh) in faces:
                _x1, _y1, _x2, _y2 = x+fx, y+fy, x+fx+fw, y+fy+fh
                use_frame[_y1:_y2, _x1:_x2] = cv2.flip(saturated_image[_y1:_y2, _x1:_x2], 0)

                eyes = eye_cascade.detectMultiScale(person, 1.3, 5)
                for eye in eyes:
                    ex, ey, ew, eh = eye
                    _x, _y, _w, _h = x+ex, y+ey, ew, eh
                    use_frame[_y:_y+_h, _x:_x+_w] = saturated_image[_y:_y+_h, _x:_x+_w]

    # 3. Draw All AI labels
    for (x, y, x2, y2, conf, cls) in boxes:
        x, y = int(x), int(y)
        name = model.names[int(cls)]
        name = 'dumbass' if name == 'person' else name
        draw_ai_label(use_frame, name, x, y, -1, conf)

    return use_frame

def main(filter_names=None):
    if filter_names is None:
        filter_names = []

    print(torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using " + str(device))

    # Image scanning model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5x')
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
                # Pre-scan frame
                results = model(frame)
                boxes = results.xyxy[0].cpu().numpy()

                frame_filtered = frame
                for filter_name in filter_names:
                    # Switch desired filter name
                    if filter_name == "ai":
                        frame_filtered, sightings = ai_filter(frame_filtered, boxes, model)
                    elif filter_name == "deepfry":
                        frame_filtered = deepfry_filter(frame_filtered, boxes, model)

                # Show window debug
                half_size_frame = cv2.resize(frame_filtered, (0, 0), fx=0.4, fy=0.4)
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
    main(["deepfry"])
