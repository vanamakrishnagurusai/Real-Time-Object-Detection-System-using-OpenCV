import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np

# -------- Optional sample file from this session (you can ignore/remove) ----------
sample_image_path = "/mnt/data/c8b54963-d25d-4cb4-ac1d-647d333113a0.png"
# ----------------------------------------------------------------------------------

# -------------------- Utility: safe file check --------------------
def check_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return path

def clean_path(p: str) -> str:
    # remove wrapping quotes and spaces from drag & drop
    return p.strip().strip('"').strip("'")

# -------------------- Load Models (lazy load on first use) --------------------
_models = {"yolo": None, "ssd": None}
_yolo_meta = {}
_ssd_meta = {}

def load_yolo():
    if _models["yolo"] is not None:
        return
    y_weights = check_file(os.path.join("yolo", "yolov4.weights"))
    y_cfg = check_file(os.path.join("yolo", "yolov4.cfg"))
    y_names = check_file(os.path.join("yolo", "coco.names"))

    net = cv2.dnn.readNet(y_weights, y_cfg)
    with open(y_names, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layers = net.getLayerNames()
    output_layers = [layers[i - 1] for i in net.getUnconnectedOutLayers()]

    _models["yolo"] = net
    _yolo_meta["classes"] = classes
    _yolo_meta["outputs"] = output_layers

def load_ssd():
    if _models["ssd"] is not None:
        return
    ptxt = check_file(os.path.join("mobilenet", "MobileNetSSD_deploy.prototxt"))
    caffemodel = check_file(os.path.join("mobilenet", "MobileNetSSD_deploy.caffemodel"))

    net = cv2.dnn.readNetFromCaffe(ptxt, caffemodel)
    classes = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person",
               "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    _models["ssd"] = net
    _ssd_meta["classes"] = classes

# -------------------- Detection functions (same logic as before) --------------------
def detect_yolo_frame(img, conf_threshold=0.5, nms_threshold=0.4):
    net = _models["yolo"]
    classes = _yolo_meta["classes"]
    output_layers = _yolo_meta["outputs"]

    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

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

    if len(boxes) > 0:
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    else:
        indexes = []

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label, (max(x,0), max(y-5,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return img

def detect_ssd_frame(img, conf_threshold=0.5):
    net = _models["ssd"]
    classes = _ssd_meta["classes"]
    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            class_id = int(detections[0, 0, i, 1])
            label = classes[class_id] if class_id < len(classes) else str(class_id)
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            x1, y1, x2, y2 = box.astype("int")
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 100, 0), 2)
            cv2.putText(img, label, (max(x1,0), max(y1-5,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,100,0), 2)
    return img

# -------------------- Runner for sources --------------------
STOP_EVENT = threading.Event()

def run_on_image(path, model_key):
    try:
        if model_key == "yolo":
            load_yolo()
        else:
            load_ssd()
    except FileNotFoundError as e:
        messagebox.showerror("Model file error", str(e))
        return

    path = clean_path(path)
    if not os.path.exists(path):
        messagebox.showerror("File error", f"Image not found:\n{path}")
        return

    img = cv2.imread(path)
    if img is None:
        messagebox.showerror("Read error", f"Could not read image:\n{path}")
        return

    out = detect_yolo_frame(img.copy()) if model_key == "yolo" else detect_ssd_frame(img.copy())
    out = cv2.resize(out, (1280, 720))
    cv2.imshow("Output", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_on_video_or_webcam(source, model_key):
    try:
        if model_key == "yolo":
            load_yolo()
        else:
            load_ssd()
    except FileNotFoundError as e:
        messagebox.showerror("Model file error", str(e))
        return

    # choose webcam or file
    cap = cv2.VideoCapture(0 if source == "webcam" else source)
    if not cap.isOpened():
        messagebox.showerror("Capture error", f"Unable to open: {source}")
        return

    STOP_EVENT.clear()
    while not STOP_EVENT.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        out = detect_yolo_frame(frame.copy()) if model_key == "yolo" else detect_ssd_frame(frame.copy())
        out = cv2.resize(out, (1280, 720))
        cv2.imshow("Output", out)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# -------------------- Thread wrapper --------------------
def start_detection_thread(source, model_key):
    # source: "webcam" or filepath (image/video)
    # decide which runner to use
    if source == "webcam" or str(source).lower().endswith(".mp4"):
        worker = threading.Thread(target=run_on_video_or_webcam, args=(source, model_key), daemon=True)
    else:
        worker = threading.Thread(target=run_on_image, args=(source, model_key), daemon=True)
    worker.start()

# -------------------- TKINTER GUI --------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Object Detection — GUI (YOLOv4 + MobileNet-SSD)")
        self.geometry("520x220")
        self.resizable(False, False)

        # model selector
        ttk.Label(self, text="Select Model:").place(x=12, y=12)
        self.model_var = tk.StringVar(value="yolo")
        self.model_combo = ttk.Combobox(self, textvariable=self.model_var, values=["yolo", "ssd"], state="readonly")
        self.model_combo.place(x=100, y=12, width=120)
        self.model_combo.set("yolo")

        # file entry + buttons
        ttk.Label(self, text="File / Image / Video:").place(x=12, y=52)
        self.path_var = tk.StringVar()
        self.path_entry = ttk.Entry(self, textvariable=self.path_var)
        self.path_entry.place(x=12, y=76, width=380)

        btn_browse = ttk.Button(self, text="Browse Image", command=self.browse_image)
        btn_browse.place(x=405, y=72, width=100)

        btn_browse_v = ttk.Button(self, text="Browse Video", command=self.browse_video)
        btn_browse_v.place(x=405, y=102, width=100)

        # action buttons
        btn_image = ttk.Button(self, text="Detect Image", command=self.detect_image)
        btn_image.place(x=12, y=132, width=120)

        btn_video = ttk.Button(self, text="Detect Video", command=self.detect_video)
        btn_video.place(x=150, y=132, width=120)

        btn_webcam = ttk.Button(self, text="Open Webcam", command=self.detect_webcam)
        btn_webcam.place(x=288, y=132, width=120)

        btn_stop = ttk.Button(self, text="Stop", command=self.stop_detection)
        btn_stop.place(x=420, y=132, width=85)

        # quick test sample button
        ttk.Button(self, text="Sample Image", command=self.use_sample).place(x=12, y=170, width=120)
        ttk.Label(self, text="(uses uploaded sample)").place(x=140, y=170)

    def browse_image(self):
        p = filedialog.askopenfilename(title="Select image", filetypes=[("Image files","*.jpg;*.jpeg;*.png;*.bmp;*.webp"), ("All files","*.*")])
        if p:
            self.path_var.set(p)

    def browse_video(self):
        p = filedialog.askopenfilename(title="Select video", filetypes=[("MP4 files","*.mp4;*.avi;*.mkv"), ("All files","*.*")])
        if p:
            self.path_var.set(p)

    def detect_image(self):
        p = clean_path(self.path_var.get())
        if not p:
            messagebox.showinfo("Use file", "Please select an image or enter a path.")
            return
        start_detection_thread(p, self.model_var.get())

    def detect_video(self):
        p = clean_path(self.path_var.get())
        if not p:
            messagebox.showinfo("Use file", "Please select a video or enter a path.")
            return
        start_detection_thread(p, self.model_var.get())

    def detect_webcam(self):
        start_detection_thread("webcam", self.model_var.get())

    def stop_detection(self):
        STOP_EVENT.set()
        # ensure OpenCV windows close
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    def use_sample(self):
        if os.path.exists(sample_image_path):
            self.path_var.set(sample_image_path)
        else:
            messagebox.showinfo("No sample", "Sample image not available in container.")

if __name__ == "__main__":
    app = App()
    app.mainloop()
