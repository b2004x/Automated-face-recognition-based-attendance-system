import cv2
import av
import numpy as np
import sqlite3
from datetime import datetime, date
from threading import Lock
import csv, io

from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from zoneinfo import ZoneInfo
import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Conv2D, Dropout, GlobalAveragePooling2D, Dense

IMG_WIDTH, IMG_HEIGHT = 224, 224
THRESHOLD = 0.5  
IMAGE_PATH = "D:\Face_detection\demo-2\z7502212317107_4285c8e68b90e68d87a69a123762538c.jpg"

model_path = r"D:\Face_detection\model\mobilenetv2-best.hdf5"
pretrain_net = MobileNetV2(
    input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
    include_top=False,
    weights=None   # ⚠️ KHÔNG dùng imagenet ở đây
)

x = pretrain_net.output
x = Conv2D(32, (3, 3), activation='relu')(x)
x = Dropout(0.2, name='extra_dropout1')(x)
x = GlobalAveragePooling2D()(x)
x = Dense(1, activation='sigmoid', name='classifier')(x)

model = Model(pretrain_net.input, x, name="mobilenetv2_spoof")
model.load_weights(model_path)

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"❌ Không đọc được ảnh: {img_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT))

    img_resized = img_resized.astype(np.float32)
    img_resized = preprocess_input(img_resized)

    img_resized = np.expand_dims(img_resized, axis=0)
    return img_resized, img

# ================= PREDICT =================
x, original_img = preprocess_image(IMAGE_PATH)

score = model.predict(x, verbose=0)[0][0]

label = "REAL" if score >= THRESHOLD else "FAKE"
color = (0, 255, 0) if label == "REAL" else (0, 0, 255)

print(f"🔍 Prediction score: {score:.4f} -> {label}")

# ================= SHOW RESULT =================
cv2.putText(
    original_img,
    f"{label}: {score:.2f}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.2,
    color,
    3
)

cv2.imshow("Anti-Spoof Test", original_img)
cv2.waitKey(0)
cv2.destroyAllWindows()