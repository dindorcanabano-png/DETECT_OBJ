import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
from twilio.rest import Client
import av
import cv2
import numpy as np
import time

# -------------------------------
# TWILIO SETTINGS
# -------------------------------
# Get these from your Twilio account
# https://www.twilio.com/console
TWILIO_SID = "YOUR_TWILIO_ACCOUNT_SID"
TWILIO_AUTH_TOKEN = "YOUR_TWILIO_AUTH_TOKEN"
TWILIO_PHONE = "+1234567890"   # Twilio number
TO_PHONE = "+639XXXXXXXXX"     # Your number

client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

# -------------------------------
# Load YOLO model once
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="YOLO Detection + Twilio Alerts")

st.title("🎥 Live Object Detection & Tracking with SMS Alert")

st.sidebar.header("⚙️ Settings")

show_boxes = st.sidebar.checkbox("Show Bounding Boxes", True)
show_labels = st.sidebar.checkbox("Show Labels", True)
show_fps = st.sidebar.checkbox("Show FPS", True)

# Choose object to alert
target_object = st.sidebar.selectbox(
    "📢 Send SMS when detected:",
    ["person", "cell phone", "car", "dog", "cat"]
)

# -------------------------------
# Video Processor
# -------------------------------
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.prev_time = time.time()
        self.last_alert_time = 0

    def send_sms_alert(self, detected_object):
        try:
            message = client.messages.create(
                body=f"⚠️ ALERT: {detected_object} detected by YOLO Camera!",
                from_=TWILIO_PHONE,
                to=TO_PHONE
            )
            print("SMS Sent:", message.sid)

        except Exception as e:
            print("Twilio Error:", e)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:

        img = frame.to_ndarray(format="bgr24")

        # YOLO tracking
        results = model.track(
            img,
            persist=True,
            conf=0.25,
            iou=0.5,
            verbose=False
        )

        annotated = img.copy()

        if results and len(results) > 0:

            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:

                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None

                for i, box in enumerate(boxes):

                    x1, y1, x2, y2 = map(int, box)

                    class_id = int(classes[i])
                    label = model.names[class_id]
                    track_id = int(ids[i]) if ids is not None else -1

                    # -------------------------------
                    # SEND TWILIO ALERT
                    # -------------------------------
                    current_time = time.time()

                    if (
                        label == target_object
                        and current_time - self.last_alert_time > 15
                    ):
                        self.send_sms_alert(label)
                        self.last_alert_time = current_time

                    color = (0, 255, 0)

                    # Draw box
                    if show_boxes:
                        cv2.rectangle(
                            annotated,
                            (x1, y1),
                            (x2, y2),
                            color,
                            2
                        )

                    # Draw label
                    if show_labels:
                        text = f"{label} ID:{track_id}" if track_id >= 0 else label

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
        # FPS DISPLAY
        # -------------------------------
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

        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24"
        )

# -------------------------------
# START WEBRTC STREAM
# -------------------------------
webrtc_streamer(
    key="yolo-clean",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
)
