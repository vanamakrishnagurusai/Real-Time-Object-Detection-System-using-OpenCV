import cv2
import numpy as np
import os

# ---------------------------------------------------
# VERIFY FILE EXISTS
# ---------------------------------------------------
def check_file(path):
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        exit()
    return path


# ---------------------------------------------------
# LOAD YOLOv4
# ---------------------------------------------------
yolo_weights = check_file("yolo/yolov4.weights")
yolo_cfg = check_file("yolo/yolov4.cfg")
yolo_names = check_file("yolo/coco.names")

yolo_net = cv2.dnn.readNet(yolo_weights, yolo_cfg)

with open(yolo_names, "r") as f:
    yolo_classes = [line.strip() for line in f.readlines()]

yolo_layers = yolo_net.getLayerNames()
yolo_output_layers = [yolo_layers[i - 1] for i in yolo_net.getUnconnectedOutLayers()]


# ---------------------------------------------------
# LOAD MobileNet-SSD
# ---------------------------------------------------
ssd_prototxt = check_file("mobilenet/MobileNetSSD_deploy.prototxt")
ssd_model = check_file("mobilenet/MobileNetSSD_deploy.caffemodel")

ssd_net = cv2.dnn.readNetFromCaffe(ssd_prototxt, ssd_model)

ssd_classes = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person",
               "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


# ---------------------------------------------------
# YOLOv4 DETECTION
# ---------------------------------------------------
def detect_yolo(img, conf_threshold=0.5, nms_threshold=0.4):
    height, width = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    outputs = yolo_net.forward(yolo_output_layers)

    class_ids, confidences, boxes = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = yolo_classes[class_ids[i]]

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return img


# ---------------------------------------------------
# SSD DETECTION
# ---------------------------------------------------
def detect_ssd(img, conf_threshold=0.5):
    height, width = img.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    ssd_net.setInput(blob)
    detections = ssd_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > conf_threshold:
            class_id = int(detections[0, 0, i, 1])
            label = ssd_classes[class_id]

            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            x1, y1, x2, y2 = box.astype("int")

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 100, 0), 2)
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

    return img


# ---------------------------------------------------
# RUN DETECTION
# ---------------------------------------------------
def run_detection(model="yolo", source="webcam"):

    # ---------------- IMAGE ----------------
    if not (source == "webcam" or source.endswith(".mp4")):

        img = cv2.imread(source)

        if img is None:
            print(f"[ERROR] Could not load image: {source}")
            return

        output = detect_yolo(img) if model == "yolo" else detect_ssd(img)

        # ✔ FIX: resize to avoid zoomed window
        output = cv2.resize(output, (1280, 720))

        cv2.imshow("Output", output)
        cv2.waitKey(0)
        return

    # ---------------- WEBCAM OR VIDEO ----------------
    cap = cv2.VideoCapture(0 if source == "webcam" else source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output = detect_yolo(frame) if model == "yolo" else detect_ssd(frame)

        # ✔ FIX: resize webcam/video output
        output = cv2.resize(output, (1280, 720))

        cv2.imshow("Output", output)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------
# MENU SYSTEM
# ---------------------------------------------------
print("\n--- MODEL SELECTION ---")
print("1. YOLOv4")
print("2. MobileNet-SSD")
choice = input("Enter 1 or 2: ")

model = "yolo" if choice == "1" else "ssd"

print("\n--- INPUT SOURCE ---")
print("1. Webcam")
print("2. Image")
print("3. Video File")
source_choice = input("Enter 1, 2, or 3: ")

if source_choice == "1":
    run_detection(model, "webcam")

elif source_choice == "2":
    filename = input("Enter image path (or drag & drop): ").strip().strip('"')
    run_detection(model, filename)

else:
    filename = input("Enter video path (or drag & drop): ").strip().strip('"')
    run_detection(model, filename)
