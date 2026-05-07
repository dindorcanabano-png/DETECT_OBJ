import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
from twilio.rest import Client
import av
import cv2
import numpy as np
import time

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="YOLO Detection + Twilio Alerts",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# =====================================================
# TWILIO
# =====================================================
account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
auth_token = st.secrets["TWILIO_AUTH_TOKEN"]

client = Client(account_sid, auth_token)

# =====================================================
# UI
# =====================================================
st.title("🎥 Live Object Detection & Tracking")

st.sidebar.header("⚙️ Settings")

show_fps = st.sidebar.checkbox("Show FPS", True)

target_object = st.sidebar.selectbox(
    "📲 Alert Object",
    ["person", "car", "dog", "cat", "cell phone"]
)

# =====================================================
# VIDEO PROCESSOR
# =====================================================
class VideoProcessor(VideoTransformerBase):

    def __init__(self):
        self.prev_time = time.time()
        self.last_alert = 0

    def send_sms(self, detected_object):
        try:
            message = client.messages.create(
                body=f"⚠️ ALERT: {detected_object} detected!",
                from_=st.secrets["TWILIO_PHONE_NUMBER"],
                to=st.secrets["YOUR_PHONE_NUMBER"]
            )
            print("SMS SENT:", message.sid)
        except Exception as e:
            print("TWILIO ERROR:", e)

    def recv(self, frame: av.VideoFrame):

        img = frame.to_ndarray(format="bgr24")

        # =================================================
        # YOLO FIXED (IMPORTANT PART)
        # =================================================
        results = model(img, conf=0.25, verbose=False)
        result = results[0]

        annotated = result.plot()

        # =================================================
        # OPTIONAL ALERT SYSTEM
        # =================================================
        if result.boxes is not None and len(result.boxes) > 0:

            classes = result.boxes.cls.cpu().numpy()

            for i in range(len(classes)):

                class_id = int(classes[i])
                label = model.names[class_id]

                current_time = time.time()

                if label == target_object and current_time - self.last_alert > 15:
                    self.send_sms(label)
                    self.last_alert = current_time

        # =================================================
        # FPS
        # =================================================
        if show_fps:
            curr = time.time()
            fps = 1 / (curr - self.prev_time)
            self.prev_time = curr

            cv2.putText(
                annotated,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# =====================================================
# STREAM
# =====================================================
webrtc_streamer(
    key="yolo-clean",
    video_processor_factory=VideoProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={
        "video": {"width": 640, "height": 480, "frameRate": 20},
        "audio": False,
    },
    async_processing=True,
)
