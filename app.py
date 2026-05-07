import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
from twilio.rest import Client
import av
import cv2
import time

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# -------------------------------
# TWILIO
# -------------------------------
account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
auth_token = st.secrets["TWILIO_AUTH_TOKEN"]

client = Client(account_sid, auth_token)
token = client.tokens.create()

# -------------------------------
# UI
# -------------------------------
st.set_page_config(
    page_title="Live Object Detection",
    layout="wide"
)

st.title("🎥 Live Object Detection")

st.sidebar.header("⚙️ Settings")

show_boxes = st.sidebar.checkbox(
    "Show Bounding Boxes",
    True
)

show_labels = st.sidebar.checkbox(
    "Show Labels",
    True
)

show_fps = st.sidebar.checkbox(
    "Show FPS",
    True
)

confidence = st.sidebar.slider(
    "Confidence",
    0.1,
    0.9,
    0.25
)

# -------------------------------
# VIDEO PROCESSOR
# -------------------------------
class VideoProcessor(VideoTransformerBase):

    def __init__(self):
        self.prev_time = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:

        img = frame.to_ndarray(format="bgr24")

        # YOLO DETECTION
        results = model.predict(
            img,
            conf=confidence,
            verbose=False
        )

        annotated = img.copy()

        if results and len(results) > 0:

            result = results[0]

            if result.boxes is not None:

                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()

                for box, cls_id, score in zip(
                    boxes,
                    classes,
                    scores
                ):

                    x1, y1, x2, y2 = map(int, box)

                    label = model.names[int(cls_id)]

                    color = (0, 255, 0)

                    # BOX
                    if show_boxes:

                        cv2.rectangle(
                            annotated,
                            (x1, y1),
                            (x2, y2),
                            color,
                            2
                        )

                    # LABEL
                    if show_labels:

                        text = f"{label} {score:.2f}"

                        cv2.putText(
                            annotated,
                            text,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                        )

        # FPS
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
# WEBRTC STREAM
# -------------------------------
webrtc_streamer(
    key="object-detect",

    video_processor_factory=VideoProcessor,

    rtc_configuration={
        "iceServers": token.ice_servers
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
