import streamlit as st

# =====================================================
# MUST BE FIRST STREAMLIT COMMAND
# =====================================================
st.set_page_config(
    page_title="YOLO Detection + Twilio Alerts",
    layout="wide"
)

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
from twilio.rest import Client
import av
import cv2
import numpy as np
import time

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# =====================================================
# TWILIO SAFE CONFIG (NO CRASH IF SECRETS MISSING)
# =====================================================
try:
    account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
    auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
    TWILIO_PHONE_NUMBER = st.secrets["TWILIO_PHONE_NUMBER"]
    YOUR_PHONE_NUMBER = st.secrets["YOUR_PHONE_NUMBER"]

    client = Client(account_sid, auth_token)
    twilio_ready = True

except Exception:
    twilio_ready = False
    client = None
    TWILIO_PHONE_NUMBER = None
    YOUR_PHONE_NUMBER = None

# =====================================================
# UI
# =====================================================
st.title("🎥 Live Object Detection & Tracking + Twilio SMS")

st.sidebar.header("⚙️ Settings")

show_boxes = st.sidebar.checkbox("Show Bounding Boxes", True)
show_labels = st.sidebar.checkbox("Show Labels", True)
show_fps = st.sidebar.checkbox("Show FPS", True)

target_object = st.sidebar.selectbox(
    "📲 SMS Alert Object",
    ["person", "car", "dog", "cat", "cell phone"]
)

st.sidebar.warning(
    "⚠️ Twilio Status: "
    + ("READY" if twilio_ready else "NOT CONFIGURED")
)

# =====================================================
# VIDEO PROCESSOR
# =====================================================
class VideoProcessor(VideoTransformerBase):

    def __init__(self):
        self.prev_time = time.time()
        self.last_alert_time = 0

    # -------------------------------
    # SMS FUNCTION
    # -------------------------------
    def send_sms(self, label):

        if not twilio_ready:
            print("Twilio not configured")
            return

        try:
            message = client.messages.create(
                body=f"⚠️ ALERT: {label} detected!",
                from_=TWILIO_PHONE_NUMBER,
                to=YOUR_PHONE_NUMBER
            )
            print("SMS SENT:", message.sid)

        except Exception as e:
            print("TWILIO ERROR:", e)

    # -------------------------------
    # FRAME PROCESSING
    # -------------------------------
    def recv(self, frame: av.VideoFrame):

        img = frame.to_ndarray(format="bgr24")

        results = model.track(
            img,
            persist=True,
            conf=0.15,
            iou=0.5,
            verbose=False
        )

        annotated = img.copy()

        if results and len(results) > 0:

            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:

                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()

                ids = (
                    result.boxes.id.cpu().numpy()
                    if result.boxes.id is not None
                    else None
                )

                for i, box in enumerate(boxes):

                    x1, y1, x2, y2 = map(int, box)

                    class_id = int(classes[i])
                    label = model.names[class_id]
                    track_id = int(ids[i]) if ids is not None else -1

                    # -------------------------
                    # SMS LOGIC (ANTI-SPAM)
                    # -------------------------
                    now = time.time()

                    if (
                        label == target_object
                        and now - self.last_alert_time > 15
                    ):
                        self.send_sms(label)
                        self.last_alert_time = now

                    color = (0, 255, 0)

                    if show_boxes:
                        cv2.rectangle(
                            annotated,
                            (x1, y1),
                            (x2, y2),
                            color,
                            2
                        )

                    if show_labels:
                        text = (
                            f"{label} ID:{track_id}"
                            if track_id >= 0
                            else label
                        )

                        cv2.putText(
                            annotated,
                            text,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                        )

        # -------------------------------
        # FPS
        # -------------------------------
        if show_fps:

            now = time.time()
            fps = 1 / (now - self.prev_time)
            self.prev_time = now

            cv2.putText(
                annotated,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24"
        )

# =====================================================
# WEBCAM STREAM
# =====================================================
webrtc_streamer(
    key="yolo-clean",

    video_processor_factory=VideoProcessor,

    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    },

    media_stream_constraints={
        "video": {
            "width": 640,
            "height": 480,
            "frameRate": 20,
        },
        "audio": False,
    },

    async_processing=True,
)
